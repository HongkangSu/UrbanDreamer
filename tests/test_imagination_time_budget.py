from pathlib import Path
from types import SimpleNamespace

import pytest

import urban_dreamer.learning.training as training
from urban_dreamer.learning.training import MainlineTrainer, resolved_imagination_budget
from urban_dreamer.workflow.config import ConfigError, ResourceConfig, load_config
from test_urban_dreamer_config import _write_config


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


def episode_trainer(monkeypatch, *, passes=2, seconds=60., durations=(42.,), previous_ema=None):
    """No model math: lock admission, completed-update accounting and elapsed time."""
    trainer = object.__new__(MainlineTrainer)
    resources = ResourceConfig(1, 1, 1, 1,
        imagination_passes_per_episode=passes, imagination_time_budget_seconds=seconds)
    trainer.config = SimpleNamespace(resources=resources)
    trainer.metrics = {'online_episodes_completed': 0.}
    if previous_ema is not None:
        trainer.metrics['imagination_pass_seconds_ema'] = previous_ema
    trainer.calls = 0
    trainer.updates = dict(actor=0, latent_critic=0, event_critic=0)
    clock = Clock()
    monkeypatch.setattr(training.time, 'perf_counter', clock)

    def imagine():
        duration = durations[min(trainer.calls, len(durations) - 1)]
        clock.now += duration
        trainer.calls += 1
        for key in trainer.updates:
            trainer.updates[key] += 1
        return dict(actor_loss=float(trainer.calls), event_critic_loss=float(trainer.calls * 2),
                    latent_critic_loss=0., imagined_events=0.)

    trainer._train_imagination_pass = imagine
    return trainer


def test_42_second_pass_is_not_repeated_with_only_18_seconds_left(monkeypatch):
    trainer = episode_trainer(monkeypatch)
    metrics = trainer._train_episode_imagination()
    assert metrics['imagination_passes_requested'] == 2
    assert metrics['imagination_passes_completed'] == 1
    assert metrics['imagination_seconds'] == 42.
    assert metrics['imagination_budget_stop_reason'] == 'insufficient_time_for_complete_pass'
    assert trainer.updates == dict(actor=1, latent_critic=1, event_critic=1)
    assert metrics['actor_loss'] == 1.


def test_two_fast_complete_passes_are_admitted_and_losses_use_completed_mean(monkeypatch):
    trainer = episode_trainer(monkeypatch, durations=(22., 24.))
    metrics = trainer._train_episode_imagination()
    assert metrics['imagination_passes_completed'] == 2
    assert metrics['imagination_seconds'] == 46.
    assert metrics['imagination_budget_stop_reason'] == 'pass_limit'
    assert metrics['actor_loss'] == 1.5
    assert metrics['event_critic_loss'] == 3.
    assert trainer.updates == dict(actor=2, latent_critic=2, event_critic=2)


def test_cold_first_pass_exceeding_target_is_finished_and_reported(monkeypatch):
    trainer = episode_trainer(monkeypatch, durations=(80.,))
    metrics = trainer._train_episode_imagination()
    assert metrics['imagination_passes_completed'] == 1
    assert metrics['imagination_seconds'] == 80.
    assert metrics['imagination_budget_stop_reason'] == 'budget_exceeded'
    assert trainer.updates == dict(actor=1, latent_critic=1, event_critic=1)


def test_disabled_time_budget_preserves_all_16_passes(monkeypatch):
    trainer = episode_trainer(monkeypatch, passes=16, seconds=0.)
    metrics = trainer._train_episode_imagination()
    assert metrics['imagination_passes_completed'] == 16
    assert metrics['imagination_seconds'] == 16 * 42.
    assert metrics['actor_loss'] == 8.5
    assert trainer.updates == dict(actor=16, latent_critic=16, event_critic=16)


def test_eight_updates_are_not_cut_off_by_historical_slow_ema(monkeypatch):
    trainer = episode_trainer(monkeypatch, passes=8, seconds=0., durations=(15.,), previous_ema=50.)
    metrics = trainer._train_episode_imagination()
    assert metrics['imagination_passes_completed'] == 8
    assert metrics['imagination_seconds'] == 120.
    assert metrics['imagination_budget_stop_reason'] == 'pass_limit'
    assert trainer.updates == dict(actor=8, latent_critic=8, event_critic=8)


def test_unexpected_slow_second_pass_is_completed_and_overrun_is_visible(monkeypatch):
    trainer = episode_trainer(monkeypatch, durations=(20., 50.))
    metrics = trainer._train_episode_imagination()
    assert metrics['imagination_passes_completed'] == 2
    assert metrics['imagination_seconds'] == 70.
    assert metrics['imagination_budget_stop_reason'] == 'budget_exceeded'
    assert trainer.updates == dict(actor=2, latent_critic=2, event_critic=2)


def test_latest_slow_pass_is_not_hidden_by_fast_historical_ema(monkeypatch):
    trainer = episode_trainer(monkeypatch, previous_ema=1., durations=(40.,))
    metrics = trainer._train_episode_imagination()
    assert metrics['imagination_pass_seconds_ema'] == 20.5
    assert metrics['imagination_passes_completed'] == 1


def test_empty_imagination_metrics_are_finite_and_still_respect_pass_limit(monkeypatch):
    trainer = episode_trainer(monkeypatch, durations=(0.,))
    trainer._train_imagination_pass = lambda: dict(actor_loss=0., imagined_events=0., latent_critic_loss=0.)
    metrics = trainer._train_episode_imagination()
    assert metrics['imagination_passes_completed'] == 2
    assert metrics['actor_loss'] == metrics['latent_critic_loss'] == metrics['imagination_seconds'] == 0.


def test_online_episode_keeps_world_updates_and_merges_budget_metrics(monkeypatch):
    trainer = episode_trainer(monkeypatch)
    trainer.validate_codec_ready = lambda: None
    trainer._set_online_prior = lambda: None
    collected = SimpleNamespace(transitions=(), total_travel_minutes=7.,
                                total_vehicle_count=0, arrival_times={}, metrics={'sample': 3.})
    trainer._run_episode = lambda **kwargs: collected
    trainer.online_replay = SimpleNamespace(append_episode=lambda *args, **kwargs: None)
    world_updates = []
    trainer._update_world = lambda *args, **kwargs: (world_updates.append(kwargs) or {'world_model_loss': 2.})
    result = trainer.train_online_episode(0)
    assert len(world_updates) == 4
    assert result['complete'] and result['imagination_passes_completed'] == 1
    assert result['sample'] == 3. and result['world_model_loss'] == 2.
    assert trainer.metrics['online_episodes_completed'] == 1.
    assert trainer.metrics['imagination_budget_stop_reason'] == 'insufficient_time_for_complete_pass'


def test_explicit_legacy_defaults_retain_historical_semantic_identity(tmp_path):
    default_path = _write_config(tmp_path / 'legacy.toml')
    explicit_path = _write_config(tmp_path / 'explicit.toml', extra='\n'
        'imagination_passes_per_episode = 16\nimagination_time_budget_seconds = 0.0\n')
    before, after = load_config(default_path), load_config(explicit_path)
    assert before.semantic_fingerprint == after.semantic_fingerprint
    assert resolved_imagination_budget(before) == dict(world_model_updates_per_episode=4,
        imagination_passes_per_episode=16, imagination_time_budget_seconds=0.,
        rollouts_per_pass=8, macro_windows_per_rollout=8)
    bounded = load_config(_write_config(tmp_path / 'bounded.toml', extra='\n'
        'imagination_passes_per_episode = 2\nimagination_time_budget_seconds = 60.0\n'))
    assert bounded.semantic_fingerprint != before.semantic_fingerprint


@pytest.mark.parametrize('setting', [
    'imagination_passes_per_episode = 0', 'imagination_passes_per_episode = true',
    'imagination_passes_per_episode = 1.5', 'imagination_time_budget_seconds = -1',
    'imagination_time_budget_seconds = nan', 'imagination_time_budget_seconds = inf',
    'imagination_time_budget_seconds = true'])
def test_budget_values_are_strictly_validated(tmp_path, setting):
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path / 'invalid.toml', extra='\n' + setting + '\n'))


def test_shipped_presets_keep_ingo_four_updates_and_other_time_budgets():
    root = Path(__file__).resolve().parents[1]
    for path in (root / 'configs').glob('*.toml'):
        config = load_config(path)
        ingo = path.name in {'ingolstadt_custom.toml', 'ingolstadt_custom_offline.toml',
                            'ingolstadt_actor_warmstart.toml'}
        assert config.resources.imagination_passes_per_episode == (4 if ingo else 2), path
        assert config.resources.imagination_time_budget_seconds == (0. if ingo else 60.), path
