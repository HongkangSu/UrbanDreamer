"""Fresh route defaults preserve explicit and historical next-hop identities."""
from dataclasses import MISSING
from pathlib import Path
import tomllib

import pytest

from test_urban_dreamer_config import _write_config
from tools.run_complete_pretrain import export_run_config
from urban_dreamer.traffic.simulation_settings import physics_identity
from urban_dreamer.workflow.config import ConfigError, load_config


def configured(path, setting=''):
    _write_config(path)
    path.write_text(path.read_text().replace('[simulation]', '[simulation]\n' + setting))
    return load_config(path)


def test_fresh_default_and_dataclass_use_current_to_dest(tmp_path):
    omitted = configured(tmp_path / 'fresh.toml')
    explicit = configured(tmp_path / 'route.toml', 'routing_mode="current_to_dest"')
    assert omitted.simulation.routing_mode == 'current_to_dest'
    assert omitted.semantic_fingerprint == explicit.semantic_fingerprint
    assert omitted.resolved_snapshot()['simulation']['routing_mode'] == 'current_to_dest'
    values = {name: getattr(omitted.simulation, name)
              for name, spec in type(omitted.simulation).__dataclass_fields__.items()
              if spec.default is MISSING and spec.default_factory is MISSING}
    assert type(omitted.simulation)(**values).routing_mode == 'current_to_dest'


def test_explicit_next_hop_keeps_prechange_fingerprint_and_snapshot(tmp_path):
    explicit = configured(tmp_path / 'new.toml', 'routing_mode="next_hop"')
    # Recorded from d255ac3, before the optional routing setting existed.
    expected = '1eace275bc3805321e9c71b53a5e20beb0fd2cc99f5057168858590329ec94a5'
    assert explicit.semantic_fingerprint == expected
    assert explicit.simulation.routing_mode == 'next_hop'
    assert 'routing_mode' not in explicit.resolved_snapshot()['simulation']


@pytest.mark.parametrize('filename', ['run_config.toml', 'source_config.toml'])
def test_legacy_frozen_snapshot_omission_keeps_next_hop(tmp_path, filename):
    legacy = configured(tmp_path / filename)
    explicit = configured(tmp_path / 'explicit.toml', 'routing_mode="next_hop"')
    assert legacy.simulation.routing_mode == 'next_hop'
    assert legacy.semantic_fingerprint == explicit.semantic_fingerprint
    assert legacy.resolved_snapshot() == explicit.resolved_snapshot()


@pytest.mark.parametrize('filename', ['run_config.toml', 'source_config.toml'])
def test_frozen_snapshot_explicit_route_setting_wins(tmp_path, filename):
    explicit = configured(tmp_path / filename, 'routing_mode="current_to_dest"')
    assert explicit.simulation.routing_mode == 'current_to_dest'


def test_route_export_preserves_control_mode_and_physics(tmp_path):
    previous = configured(tmp_path / 'old.toml', 'routing_mode="next_hop"')
    target = configured(tmp_path / 'route.toml', 'routing_mode="current_to_dest"')
    assert target.semantic_fingerprint != previous.semantic_fingerprint
    assert physics_identity(target.simulation) == physics_identity(previous.simulation)
    path = tmp_path / 'exported.toml'
    export_run_config(target, path)
    restored = load_config(path)
    assert restored.simulation.routing_mode == 'current_to_dest'
    assert restored.semantic_fingerprint == target.semantic_fingerprint


@pytest.mark.parametrize('mode', ['next_hop', 'current_to_dest'])
@pytest.mark.parametrize('filename', ['arbitrary.toml', 'run_config.toml', 'source_config.toml'])
def test_new_exports_explicitly_round_trip_each_mode(tmp_path, mode, filename):
    config = configured(tmp_path / 'input.toml', f'routing_mode="{mode}"')
    before = config.resolved_snapshot()
    path = tmp_path / filename
    export_run_config(config, path)
    assert tomllib.loads(path.read_text())['simulation']['routing_mode'] == mode
    restored = load_config(path)
    assert restored.simulation.routing_mode == mode
    assert restored.semantic_fingerprint == config.semantic_fingerprint
    assert restored.resolved_snapshot() == before
    assert config.resolved_snapshot() == before


def test_every_shipped_profile_explicitly_selects_current_to_dest():
    paths = list((Path(__file__).resolve().parents[1] / 'configs').glob('*.toml'))
    assert paths
    for path in paths:
        assert tomllib.loads(path.read_text())['simulation']['routing_mode'] == 'current_to_dest', path
        assert load_config(path).simulation.routing_mode == 'current_to_dest', path


def test_explicit_fp32_keeps_prechange_checkpoint_fingerprint(tmp_path):
    path = _write_config(tmp_path / 'fp32.toml', routing_mode='next_hop')
    path.write_text(path.read_text().replace('[resources]',
        '[resources]\nworld_model_imagination_precision="fp32"'))
    # Independently reproduced using the unchanged d255ac3 source snapshot.
    assert load_config(path).semantic_fingerprint == (
        'a39f77c4f1b0f775f59973609322cbe878a42ba8b3d6ba339629e14e379fefb5')


@pytest.mark.parametrize('value', ['"route"', 'true', '1', '[]', '{}'])
def test_invalid_routing_mode_is_rejected(tmp_path, value):
    with pytest.raises(ConfigError, match='routing_mode'):
        configured(tmp_path / 'config.toml', 'routing_mode=' + value)
