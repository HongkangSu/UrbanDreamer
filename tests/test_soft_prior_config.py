"""New background-prior profiles must not relabel archived hard checkpoints."""
import ast
from dataclasses import replace
from pathlib import Path

import pytest

from test_urban_dreamer_config import _write_config
from urban_dreamer.workflow.config import ConfigError, MethodConfig, _semantic_fingerprint, load_config


def fingerprint(config, method):
    values = {name: getattr(config, name) for name in
              ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}
    values['method'] = method
    return _semantic_fingerprint(**values)


def test_archived_omitted_fields_keep_hard_action_conditioning(tmp_path):
    config = load_config(_write_config(tmp_path/'old.toml'))
    assert config.method.prior_filter == 'hard'
    assert config.method.prior_action_conditioned is True
    assert config.method.prior_filter_input is False
    assert fingerprint(config, replace(config.method, prior_filter_strength=99.)) == config.semantic_fingerprint
    assert fingerprint(config, replace(config.method, prior_filter='soft')) != config.semantic_fingerprint
    assert fingerprint(config, replace(config.method, prior_action_conditioned=False)) != config.semantic_fingerprint
    assert fingerprint(config, replace(config.method, prior_filter_input=True)) != config.semantic_fingerprint


def test_explicit_soft_profile_defaults_to_state_only_output_filter(tmp_path):
    config = load_config(_write_config(tmp_path/'soft.toml', extra='\n[method]\nprior_filter="soft"\n'))
    assert config.method.prior_filter == 'soft'
    assert config.method.prior_filter_strength == .1
    assert config.method.prior_action_conditioned is False
    assert config.method.prior_filter_input is False
    assert fingerprint(config, replace(config.method, prior_filter_strength=20.)) != config.semantic_fingerprint
    assert MethodConfig().prior_filter == 'soft'
    assert MethodConfig().prior_filter_strength == .1


@pytest.mark.parametrize('entry', ['prior_filter="invalid"', 'prior_filter_strength=-1',
    'prior_filter_strength=nan', 'prior_action_conditioned=1', 'prior_filter_input="true"'])
def test_invalid_prior_profiles_fail_before_model_creation(tmp_path, entry):
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path/'bad.toml', extra='\n[method]\n'+entry+'\n'))


def test_repository_profiles_explicitly_select_output_only_soft_background():
    root = Path(__file__).resolve().parents[1]
    profiles = [load_config(path) for path in (root/'configs').glob('*.toml')
                if '[method]' in path.read_text()]
    assert profiles
    for config in profiles:
        assert config.method.prior_filter == 'soft'
        assert config.method.prior_filter_input is False
        assert config.method.prior_action_conditioned is False


def test_config_driven_world_model_tools_forward_prior_contract():
    root = Path(__file__).resolve().parents[1]
    checked = 0
    for path in (root/'tools').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', '')
            if name not in ('UrbanDreamerWorldModel', 'Dynamics'):
                continue
            rank = next((k for k in node.keywords if k.arg == 'low_frequency_rank'), None)
            if rank is None or not isinstance(rank.value, ast.Attribute):
                continue
            if not ast.unparse(rank.value.value).endswith('.method'):
                continue
            keys = {k.arg for k in node.keywords}
            assert {'prior_filter', 'prior_filter_strength', 'prior_action_conditioned', 'prior_filter_input'} <= keys, path
            checked += 1
    assert checked >= 14
