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
        argv += ['--no-global-reward-only', '--reward-objective', 'legacy_exposure']
    monkeypatch.setattr(tool, 'load_config', lambda path: object())
    with pytest.raises(CapturedArgs):
        tool.main(argv)
    assert seen[0].global_reward_only is not legacy
    assert seen[0].reward_objective == ('legacy_exposure' if legacy else 'observed_mean')


def test_launcher_defaults_align_all_methods_and_legacy_is_explicit(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(launcher, 'launch', lambda args: seen.append(args))
    for legacy in (False, True):
        methods = list(launcher.METHODS) + ([] if legacy else list(launcher.EXTRA_METHODS))
        argv = ['launch', '--output-root', str(tmp_path / 'run'), '--methods', *methods]
        if legacy:
            argv += ['--no-global-reward-only', '--reward-objective', 'legacy_exposure']
        launcher.main(argv)
        args = seen[-1]
        assert args.global_reward_only is not legacy
        runs = launcher.build_runs(tmp_path / 'run', tmp_path / 'source', args, 'repair')
        flag = '--no-global-reward-only' if legacy else '--global-reward-only'
        assert all(flag in run['command'] for run in runs)
    args.methods = list(launcher.EXTRA_METHODS)
    runs = launcher.build_runs(tmp_path / 'run', tmp_path / 'source', args, 'explicit_legacy')
    assert all(run['command'][run['command'].index('--reward-objective') + 1] ==
               'legacy_exposure' for run in runs)


@pytest.mark.parametrize('tool,method', [(ac, None), (baseline, 'dreamer'), (baseline, 'ippo')])
def test_observed_mean_rejects_implicit_timeout_shaping(tool, method, tmp_path):
    argv = ['--output-dir', str(tmp_path / 'run'), '--no-global-reward-only']
    if method:
        argv.extend(['--method', method])
    with pytest.raises(SystemExit) as error:
        tool.main(argv)
    assert error.value.code == 2


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
