"""Entropy annealing must use run progress, independent of event population."""
from dataclasses import replace

import pytest

from urban_dreamer.learning.entropy_schedule import entropy_schedule_profile
from urban_dreamer.workflow.config import ConfigError, MethodConfig, load_config
from test_urban_dreamer_config import _write_config


def profile(completed, total=5000, **settings):
    return entropy_schedule_profile(MethodConfig(actor_entropy_mode='annealed', **settings),
        completed_episodes=completed, total_episodes=total)


@pytest.mark.parametrize('completed,expected', [
    (0, .002), (499, .002), (500, .002), (2250, .00105),
    (4000, .0001), (5000, .0001), (6000, .0001),
])
def test_default_schedule_hold_cosine_and_floor(completed, expected):
    result = profile(completed)
    assert result['coefficient'] == pytest.approx(expected)
    assert result['mode'] == 'annealed'
    assert result['completed_episodes'] == completed
    assert result['total_episodes'] == 5000
    assert result['progress'] == min(completed / 5000, 1.)


def test_schedule_is_monotone_and_independent_of_budget_length():
    coefficients = [profile(episode, 100)['coefficient'] for episode in range(101)]
    assert all(left >= right for left, right in zip(coefficients, coefficients[1:]))
    assert profile(2500)['coefficient'] == profile(50, 100)['coefficient']


def test_restored_episode_count_keeps_decay_progress():
    restored_count = 2250
    assert profile(restored_count)['coefficient'] == pytest.approx(.00105)
    assert profile(restored_count + 1)['coefficient'] < profile(restored_count)['coefficient']


@pytest.mark.parametrize('mode,coefficient', [('legacy', .002), ('population', .002), ('off', 0.)])
def test_existing_modes_retain_historical_base_coefficient(mode, coefficient):
    config = MethodConfig(actor_entropy_mode=mode)
    for completed in (0, 1000, 5000):
        result = entropy_schedule_profile(config, completed_episodes=completed, total_episodes=5000)
        assert result['coefficient'] == coefficient


@pytest.mark.parametrize('completed,total', [(-1, 10), (0, 0), (0, -1), (.5, 10),
                                          (0, 1.5), (True, 10), (0, True)])
def test_invalid_episode_counts_rejected(completed, total):
    with pytest.raises(ValueError, match='episodes'):
        profile(completed, total)


@pytest.mark.parametrize('settings', [
    {'actor_entropy_initial_coefficient': float('nan')},
    {'actor_entropy_final_coefficient': float('inf')},
    {'actor_entropy_initial_coefficient': -.001},
    {'actor_entropy_final_coefficient': -.0001},
    {'actor_entropy_final_coefficient': .003},
    {'actor_entropy_hold_fraction': -.1},
    {'actor_entropy_hold_fraction': .8},
    {'actor_entropy_end_fraction': 1.1},
    {'actor_entropy_end_fraction': True},
])
def test_invalid_schedule_rejected_even_with_direct_method_config(settings):
    with pytest.raises(ValueError, match='entropy'):
        profile(1000, **settings)


def test_custom_zero_floor_and_immediate_decay_supported():
    assert profile(0, 100, actor_entropy_hold_fraction=0.,
                   actor_entropy_end_fraction=1., actor_entropy_final_coefficient=0.)['coefficient'] == .002
    assert profile(100, 100, actor_entropy_hold_fraction=0.,
                   actor_entropy_end_fraction=1., actor_entropy_final_coefficient=0.)['coefficient'] == 0.


@pytest.mark.parametrize('mode', ['legacy', 'population', 'off'])
def test_new_inactive_settings_preserve_old_fingerprint_bytes(tmp_path, monkeypatch, mode):
    import urban_dreamer.workflow.config as module

    config = load_config(_write_config(tmp_path / 'old.toml',
        extra=f'\n[method]\nactor_entropy_mode = "{mode}"\n'))
    original_asdict = module.asdict

    def old_asdict(value):
        result = original_asdict(value)
        if isinstance(value, module.MethodConfig):
            for field in tuple(result):
                if field.startswith('actor_entropy_') and field != 'actor_entropy_mode':
                    result.pop(field)
        return result

    monkeypatch.setattr(module, 'asdict', old_asdict)
    old_hash = module._semantic_fingerprint(**{key: getattr(config, key) for key in
        ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')})
    assert old_hash == config.semantic_fingerprint


def test_annealed_schedule_settings_are_checkpoint_identity(tmp_path):
    config = load_config(_write_config(tmp_path / 'new.toml',
        extra='\n[method]\nactor_entropy_mode = "annealed"\n'))
    import urban_dreamer.workflow.config as module

    for name, value in [('actor_entropy_initial_coefficient', .003),
                        ('actor_entropy_final_coefficient', .0002),
                        ('actor_entropy_hold_fraction', .2),
                        ('actor_entropy_end_fraction', .9)]:
        changed = replace(config.method, **{name: value})
        digest = module._semantic_fingerprint(**{
            key: changed if key == 'method' else getattr(config, key)
            for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')})
        assert digest != config.semantic_fingerprint


@pytest.mark.parametrize('setting,value', [
    ('actor_entropy_initial_coefficient', 'nan'),
    ('actor_entropy_final_coefficient', '0.003'),
    ('actor_entropy_final_coefficient', '-0.001'),
    ('actor_entropy_hold_fraction', '0.8'),
    ('actor_entropy_end_fraction', '1.1'),
    ('actor_entropy_end_fraction', 'true'),
])
def test_invalid_schedule_toml_rejected(tmp_path, setting, value):
    with pytest.raises(ConfigError, match='entropy'):
        load_config(_write_config(tmp_path / 'bad.toml', extra=
            f'\n[method]\nactor_entropy_mode = "annealed"\n{setting} = {value}\n'))
