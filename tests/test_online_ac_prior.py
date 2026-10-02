"""Behavior-prior protocol tests for the real-SUMO Online AC runner."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / 'tools/train_toy_actor_critic.py'


def module():
    spec = importlib.util.spec_from_file_location('online_ac_prior_runner', PATH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def args(tmp_path, weight=1.):
    return SimpleNamespace(config=PATH.parents[1] / 'configs/toy_cross_bottleneck.toml',
        output_dir=tmp_path, episodes=1, eval_every=1, eval_seeds=1, seed=42, device='cpu',
        threads=1, actor_lr=2e-4, global_critic_lr=3e-4, vehicle_limit=2,
        session='microstep', step_seconds=1., backend='traci', resume=False,
        sp_prior_weight=weight, sp_prior_temperature=30., sp_prior_hold_episodes=200,
        sp_prior_off_episode=1000)


def test_paired_evaluation_reuses_seeds_labels_and_restores_prior(tmp_path):
    tool = module()
    runner = SimpleNamespace(args=args(tmp_path), actor=tool.NextHopActor())
    tool.set_episode_prior(runner, 600)
    calls = []
    def episode(**kwargs):
        calls.append((kwargs, runner.actor.prior_weight))
        return [], dict(policy=kwargs['policy'], seed=kwargs['seed'], selection_score=-1.)
    runner.run_episode = episode
    records = tool.evaluate(runner, tmp_path, episode=200, seeds=[42, 43], phase='evaluation')
    assert [c[0] for c in calls] == [dict(policy='deterministic', seed=s) for s in [42, 43, 42, 43]]
    assert [c[1] for c in calls] == [1., 1., 0., 0.]
    assert [r['policy'] for r in records] == ['deterministic'] * 2 + ['deterministic_no_prior'] * 2
    assert runner.actor.prior_weight == pytest.approx(.5)
    assert records[-1]['sp_prior_temperature'] == 30.


def test_evaluation_restores_prior_on_failure(tmp_path):
    tool = module()
    runner = SimpleNamespace(args=args(tmp_path), actor=tool.NextHopActor())
    tool.set_episode_prior(runner, 600)
    def fail(**kwargs):
        raise RuntimeError('evaluation failure')
    runner.run_episode = fail
    with pytest.raises(RuntimeError, match='evaluation failure'):
        tool.evaluate(runner, tmp_path, episode=200, seeds=[42], phase='evaluation')
    assert runner.actor.prior_weight == pytest.approx(.5)


def test_resume_restores_schedule_and_rejects_changed_profile(tmp_path):
    tool = module()
    runner = tool.RealActorCritic(args(tmp_path))
    tool.set_episode_prior(runner, 600)
    path = tmp_path / 'checkpoint.pt'
    runner.save(path, 600, 1.)
    restored = tool.RealActorCritic(args(tmp_path))
    assert restored.load(path) == (601, 1.)
    assert restored.actor.prior_weight == pytest.approx(.5)
    restored.args.sp_prior_temperature = 31.
    with pytest.raises(ValueError, match='prior protocol'):
        restored.load(path)


def test_legacy_checkpoint_is_compatible_only_without_prior(tmp_path):
    tool = module()
    runner = tool.RealActorCritic(args(tmp_path, weight=0.))
    path = tmp_path / 'legacy.pt'
    runner.save(path, 17, 1.)
    state = torch.load(path, weights_only=False)
    for key in tool.PRIOR_DEFAULTS:
        state['args'].pop(key)
    torch.save(state, path)
    assert runner.load(path) == (18, 1.)
    runner.args.sp_prior_weight = 1.
    with pytest.raises(ValueError, match='prior protocol'):
        runner.load(path)


def test_training_and_update_share_schedule_and_bare_actor_ranks_best(tmp_path, monkeypatch):
    tool = module()
    observed = []
    def episode(self, **kwargs):
        if kwargs.get('collect'):
            observed.append(('collect', self.actor.prior_weight))
        return [], dict(policy=kwargs['policy'], selection_score=-1. if self.actor.prior_weight == 0 else -100.,
                        **tool.actor_prior_metrics(self.actor))
    def update(self, *unused):
        observed.append(('update', self.actor.prior_weight))
        return {}
    monkeypatch.setattr(tool.RealActorCritic, 'run_episode', episode)
    monkeypatch.setattr(tool.RealActorCritic, 'update', update)
    assert tool.main(['--output-dir', str(tmp_path), '--episodes', '2', '--vehicle-limit', '2',
        '--skip-initial-evaluation', '--skip-reference-baselines', '--sp-prior-weight', '1',
        '--sp-prior-hold-episodes', '0', '--sp-prior-off-episode', '2']) == 0
    assert observed == [('collect', .5), ('update', .5), ('collect', 0.), ('update', 0.)]
    state = torch.load(tmp_path / 'best.pt', weights_only=False)
    assert state['best'] == 1.
    records = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert [r['sp_prior_weight'] for r in records if r['phase'] == 'train'] == [.5, 0.]
    assert (tmp_path / 'initial.pt').exists()
    protocol = json.loads((tmp_path / 'protocol.json').read_text())
    assert protocol['shortest_path_prior']['checkpoint_selection_policy'] == 'deterministic_no_prior'


def test_archive_and_metrics_keep_actual_prior(tmp_path, monkeypatch):
    from urban_dreamer.learning.episode_archive import read_episode_archive
    tool = module()
    runner = tool.RealActorCritic(args(tmp_path))
    tool.set_episode_prior(runner, 600)
    monkeypatch.setattr(tool.PoissonDemand, 'sample', lambda self, **kwargs: ())
    class EmptySession:
        done = True
        travel_times = {}
        microstep_seconds = 1.
        metrics = {'microsteps': 0}
        def __init__(self, *unused): pass
        def __enter__(self): return self
        def __exit__(self, *unused): pass
        def reset(self):
            return dict(time=0., flow=runner.network.extract_flow({}), active_vehicle_count=0, vehicles=[])
    monkeypatch.setattr(tool, 'SumoSession', EmptySession)
    path = tmp_path / 'episode.npz'
    _, metrics = runner.run_episode(policy='stochastic', seed=42, collect=True, archive_path=path,
                                   archive_metadata=dict(episode=600))
    archive = read_episode_archive(path)
    assert metrics['sp_prior_weight'] == pytest.approx(.5)
    assert archive['metadata']['sp_prior_weight'] == pytest.approx(.5)
    assert archive['metadata']['sp_prior_temperature'] == 30.
