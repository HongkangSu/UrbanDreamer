"""CLI, provenance and resume barriers for the history-constrained experiment."""
from types import SimpleNamespace

import pytest

from test_city_dataset_launcher import load_tool
from test_online_ac_prior import args as ac_args
from urban_dreamer.learning.sumo_sampling import resolve_session_factory


@pytest.mark.parametrize('name,extra', [
    ('train_toy_actor_critic', []), ('train_toy_baseline', ['--method', 'mbpo']),
    ('collect_random_episodes', []),
])
def test_cli_rejects_legacy_session(tmp_path, name, extra):
    with pytest.raises(SystemExit):
        load_tool(name).main(['--config', 'missing.toml', '--output-dir', str(tmp_path),
                             '--forbid-revisit', '--session', 'microstep', *extra])


@pytest.mark.parametrize('name,extra', [
    ('train_toy_actor_critic', []), ('train_toy_baseline', ['--method', 'dreamer']),
])
def test_cli_rejects_nonzero_prior(tmp_path, name, extra):
    with pytest.raises(SystemExit):
        load_tool(name).main(['--config', 'missing.toml', '--output-dir', str(tmp_path),
            '--forbid-revisit', '--session', 'event', '--sp-prior-weight', '1', *extra])


def test_launch_masks_all_methods_and_never_enables_prior(tmp_path):
    tool = load_tool('run_city_baseline_dataset')
    args = SimpleNamespace(cities=tool.CITIES, methods=tool.METHODS, devices=[], episodes=5000,
        seed=42, checkpoint_every=1, eval_every=100, eval_seeds=3, forbid_revisit=True,
        sp_prior_weight=0.)
    runs = tool.build_runs(tmp_path, tmp_path / 'source', args, 'no_revisit')
    assert len(runs) == 15
    for run in runs:
        assert '--forbid-revisit' in run['command']
        if run['method'] == 'random':
            assert '--sp-prior-weight' not in run['command']
        else:
            assert run['command'][run['command'].index('--sp-prior-weight') + 1] == '0.0'
    args.sp_prior_weight = 1.
    with pytest.raises(ValueError, match='zero shortest-path prior'):
        tool.build_runs(tmp_path, tmp_path / 'source', args, 'invalid')


@pytest.mark.parametrize('factory', [resolve_session_factory,
    load_tool('train_toy_baseline').resolve_session_factory])
def test_session_mode_is_opt_in(factory):
    assert 'forbid_revisit' not in factory('event', 1., 'libsumo').keywords
    assert factory('event', 1., 'libsumo', forbid_revisit=True).keywords['forbid_revisit'] is True
    with pytest.raises(ValueError, match='event'):
        factory('microstep', 1., forbid_revisit=True)


def test_baseline_resume_requires_matching_history_mode():
    tool = load_tool('train_toy_baseline')
    tool.validate_prior_resume({}, SimpleNamespace(forbid_revisit=False))
    tool.validate_prior_resume({'forbid_revisit': True}, SimpleNamespace(forbid_revisit=True))
    for saved, active in [({}, True), ({'forbid_revisit': True}, False)]:
        with pytest.raises(ValueError, match='forbid_revisit'):
            tool.validate_prior_resume(saved, SimpleNamespace(forbid_revisit=active))


def test_ac_resume_rejects_old_checkpoint_for_history_mode(tmp_path):
    tool = load_tool('train_toy_actor_critic')
    args = ac_args(tmp_path, weight=0.)
    runner = tool.RealActorCritic(args)
    checkpoint = tmp_path / 'old.pt'
    runner.save(checkpoint, 3, 0.)
    runner.args.forbid_revisit = True
    with pytest.raises(ValueError, match='forbid_revisit'):
        runner.load(checkpoint)


def test_runners_share_history_mask_and_event_session_mode(tmp_path):
    from urban_dreamer.learning.baselines.common import EpisodeRunner
    tool = load_tool('train_toy_actor_critic')
    args = ac_args(tmp_path, weight=0.)
    args.session = 'event'
    args.forbid_revisit = True
    ac = tool.RealActorCritic(args)
    assert ac.candidates.forbid_revisit
    assert ac.session_factory.keywords['forbid_revisit'] is True
    baseline = EpisodeRunner(args, session_factory=resolve_session_factory(
        args.session, args.step_seconds, args.backend, forbid_revisit=True))
    assert baseline.candidates.forbid_revisit
    assert baseline.session_factory.keywords['forbid_revisit'] is True
