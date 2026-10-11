"""New experiment entry points align team reward without relabeling history."""
from types import SimpleNamespace

import pytest

from tools import collect_random_episodes, run_city_baseline_dataset as launcher
from tools import train_toy_actor_critic as ac, train_toy_baseline as baseline


class CapturedArgs(Exception):
    pass


@pytest.mark.parametrize('tool,method', [(ac, None), (collect_random_episodes, None)] +
                         [(baseline, name) for name in ('gnac', 'dreamer', 'mbpo', 'ippo', 'mappo', 'iql', 'mamba')])
@pytest.mark.parametrize('legacy', [False, True])
def test_reward_defaults_and_explicit_legacy_mode(monkeypatch, tmp_path, tool, method, legacy):
    seen = []

    def capture(config, args):
        seen.append(args)
        raise CapturedArgs

    owner = baseline.common if tool is baseline else tool
    monkeypatch.setattr(owner, 'resolve_collection_settings', capture)
    argv = ['--output-dir', str(tmp_path / 'run'), '--config', str(tmp_path / 'unused.toml')]
    if method:
        argv += ['--method', method]
    if legacy:
        argv += ['--no-global-reward-only']
    monkeypatch.setattr(tool, 'load_config', lambda path: object())
    if legacy and method in launcher.EXTRA_METHODS:
        with pytest.raises(SystemExit):
            tool.main(argv)
        assert not seen
        return
    with pytest.raises(CapturedArgs):
        tool.main(argv)
    assert seen[0].global_reward_only is not legacy


def test_launcher_defaults_align_all_methods_and_legacy_is_explicit(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(launcher, 'launch', lambda args: seen.append(args))
    for legacy in (False, True):
        methods = list(launcher.METHODS) + ([] if legacy else list(launcher.EXTRA_METHODS))
        argv = ['launch', '--output-root', str(tmp_path / 'run'), '--methods', *methods]
        if legacy:
            argv += ['--no-global-reward-only']
        launcher.main(argv)
        args = seen[-1]
        assert args.global_reward_only is not legacy
        runs = launcher.build_runs(tmp_path / 'run', tmp_path / 'source', args, 'repair')
        flag = '--no-global-reward-only' if legacy else '--global-reward-only'
        assert all(flag in run['command'] for run in runs)
    args.methods = list(launcher.EXTRA_METHODS)
    with pytest.raises(ValueError, match='pure global'):
        launcher.build_runs(tmp_path / 'run', tmp_path / 'source', args, 'invalid')


def test_learning_options_reach_cli_without_changing_sampling_defaults(monkeypatch, tmp_path):
    seen = []

    def capture(config, args):
        seen.append(args)
        raise CapturedArgs

    monkeypatch.setattr(baseline.common, 'resolve_collection_settings', capture)
    with pytest.raises(CapturedArgs):
        baseline.main(['--method', 'iql', '--output-dir', str(tmp_path / 'run'),
                       '--iql-sample-to-insert-ratio', '1.5', '--ppo-target-kl', '.02'])
    assert seen[0].iql_sample_to_insert_ratio == 1.5
    assert seen[0].ppo_target_kl == .02
    assert seen[0].sp_prior_weight == 0.
