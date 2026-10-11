"""Mainline entropy units are an explicit, checkpointed method choice."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from test_urban_dreamer_config import _write_config
from test_urban_dreamer_training import make_trainer
from urban_dreamer.workflow.config import ConfigError, load_config


def test_omitted_entropy_preserves_historical_fingerprint_bytes(tmp_path, monkeypatch):
    import urban_dreamer.workflow.config as module

    config = load_config(_write_config(tmp_path / 'old.toml'))
    assert config.method.actor_entropy_mode == 'legacy'
    original_asdict = module.asdict

    def old_asdict(value):
        result = original_asdict(value)
        if isinstance(value, module.MethodConfig):
            result.pop('actor_entropy_mode')
        return result

    monkeypatch.setattr(module, 'asdict', old_asdict)
    old_hash = module._semantic_fingerprint(**{key: getattr(config, key) for key in
        ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')})
    assert config.semantic_fingerprint == old_hash


def test_entropy_mode_is_part_of_nonlegacy_checkpoint_identity(tmp_path):
    hashes = {}
    for mode in (None, 'legacy', 'population', 'off'):
        extra = '' if mode is None else f'\n[method]\nactor_entropy_mode = "{mode}"\n'
        config = load_config(_write_config(tmp_path / f'{mode}.toml', extra=extra))
        hashes[mode] = config.semantic_fingerprint
        assert config.resolved_snapshot()['method']['actor_entropy_mode'] == (mode or 'legacy')
    assert hashes[None] == hashes['legacy']
    assert len({hashes['legacy'], hashes['population'], hashes['off']}) == 3


@pytest.mark.parametrize('value', ['true', '1', '0.5', '""', '"automatic"', '[]'])
def test_entropy_mode_rejects_invalid_type_and_value(tmp_path, value):
    with pytest.raises(ConfigError, match='actor_entropy_mode'):
        load_config(_write_config(tmp_path / 'bad.toml',
                                  extra=f'\n[method]\nactor_entropy_mode = {value}\n'))


def test_every_shipped_config_explicitly_selects_fixed_entropy_annealing():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / 'configs').glob('*.toml'))
    assert paths
    for path in paths:
        config = load_config(path)
        assert config.method.actor_entropy_mode == 'annealed', path.name
        assert config.method.actor_entropy_final_coefficient == .0001, path.name


@pytest.mark.parametrize('spread,scale', [(.1, 1.), (3., 3.)])
def test_restored_legacy_metric_does_not_override_population_training(tmp_path, monkeypatch,
                                                                   spread, scale):
    import urban_dreamer.learning.training as module

    trainer = make_trainer(tmp_path)
    trainer.config = replace(trainer.config,
        method=replace(trainer.config.method, actor_entropy_mode='population'))
    state = trainer.state_dict()
    state['metrics']['actor_entropy_mode'] = 'legacy'
    trainer.load_state_dict(state)
    trainer.online_replay = type('Seeds', (), {
        'has_active_seeds': True, 'sample_active_seeds': lambda *args, **kwargs: None})()
    monkeypatch.setenv('URB_IMAGINATION_WORKERS', '0')
    monkeypatch.delenv('URB_ACTOR_ENTROPY_MODE', raising=False)
    conditions = []

    def imagine(*args, **kwargs):
        actor_parameter = next(trainer.actor.parameters()).flatten()[0]
        q_parameter = next(trainer.event_critic.parameters()).flatten()[0]
        value_parameter = next(trainer.latent_critic.parameters()).flatten()[0]
        global_advantage = torch.tensor([-spread, spread], requires_grad=True)
        global_return = torch.tensor([-.1, -.2], requires_grad=True)
        conditions.extend([global_advantage, global_return])
        return [dict(vehicle_id=['a', 'b'], executed_q=q_parameter.expand(2),
            global_return=global_return, global_advantage=global_advantage,
            timeout_return=torch.zeros(2), entropy=actor_parameter.sigmoid().expand(2),
            log_prob=-actor_parameter.square().expand(2), population=1000,
            counterfactual_baseline=torch.zeros(2))], value_parameter.square()

    monkeypatch.setattr(trainer, '_imagine_from_seed', imagine)
    stats = trainer._train_imagination_pass()
    assert trainer.metrics['actor_entropy_mode'] == 'population'
    assert stats['entropy_advantage_scale'] == pytest.approx(scale)
    assert stats['entropy_coefficient_mean'] == pytest.approx(module.ACTOR_ENTROPY_COEF / 1000 / scale)
    assert all(condition.grad is None for condition in conditions)
    assert all(parameter.grad is None for parameter in trainer.world_model.parameters())
    assert any(parameter.grad is not None for parameter in trainer.actor.parameters())


def test_explicit_diagnostic_environment_override_selects_entropy_mode(tmp_path, monkeypatch):
    # The execution runner rejects a production mismatch; the trainer keeps
    # this explicit override for isolated frozen-checkpoint experiments.
    import urban_dreamer.learning.training as module

    trainer = make_trainer(tmp_path)
    trainer.online_replay = type('Seeds', (), {
        'has_active_seeds': True, 'sample_active_seeds': lambda *args, **kwargs: None})()
    monkeypatch.setenv('URB_IMAGINATION_WORKERS', '0')
    monkeypatch.setenv('URB_ACTOR_ENTROPY_MODE', 'off')

    def imagine(*args, **kwargs):
        actor = next(trainer.actor.parameters()).flatten()[0]
        q = next(trainer.event_critic.parameters()).flatten()[0]
        critic = next(trainer.latent_critic.parameters()).flatten()[0]
        return [dict(vehicle_id=['a'], executed_q=q[None], global_return=torch.zeros(1),
            global_advantage=torch.ones(1), timeout_return=torch.zeros(1),
            entropy=actor.sigmoid()[None], log_prob=-actor.square()[None],
            counterfactual_baseline=torch.zeros(1))], critic.square()

    monkeypatch.setattr(trainer, '_imagine_from_seed', imagine)
    stats = trainer._train_imagination_pass()
    assert trainer.metrics['actor_entropy_mode'] == 'off'
    assert stats['entropy_coefficient_mean'] == 0.


@pytest.mark.parametrize('completed,coefficient', [(0,.002),(45,.00105),(80,.0001)])
def test_imagination_annealing_uses_real_episode_clock_without_population_scaling(tmp_path,monkeypatch,completed,coefficient):
    trainer = make_trainer(tmp_path)
    trainer.config = replace(trainer.config,
        run=replace(trainer.config.run,online_training_episodes=100),
        method=replace(trainer.config.method,actor_entropy_mode='annealed'))
    trainer.metrics['online_episodes_completed'] = completed
    trainer.online_replay = type('Seeds',(),{'has_active_seeds':True,
        'sample_active_seeds':lambda *args,**kwargs:None})()
    monkeypatch.setenv('URB_IMAGINATION_WORKERS','0')
    monkeypatch.delenv('URB_ACTOR_ENTROPY_MODE',raising=False)
    def imagine(*args,**kwargs):
        actor=next(trainer.actor.parameters()).flatten()[0]
        q=next(trainer.event_critic.parameters()).flatten()[0]
        critic=next(trainer.latent_critic.parameters()).flatten()[0]
        return [dict(vehicle_id=['a','b'],executed_q=q.expand(2),global_return=torch.zeros(2),
            global_advantage=torch.tensor([-3.,3.]),timeout_return=torch.zeros(2),
            entropy=actor.sigmoid().expand(2),log_prob=-actor.square().expand(2),
            counterfactual_baseline=torch.zeros(2),population=10000)],critic.square()
    monkeypatch.setattr(trainer,'_imagine_from_seed',imagine)
    stats=trainer._train_imagination_pass()
    assert stats['entropy_coefficient_mean']==pytest.approx(coefficient)
    assert stats['entropy_schedule_progress']==completed/100
    assert trainer.metrics['online_episodes_completed']==completed
