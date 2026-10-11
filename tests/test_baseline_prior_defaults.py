"""New learner runs disable prior by default; explicit ablations and legacy resumes stay usable."""
from types import SimpleNamespace

import pytest

from tools import run_city_baseline_dataset as city
from tools import restart_baselines_meso as restart
from tools import train_toy_actor_critic as ac
from tools import train_toy_baseline as baseline


class CapturedArgs(Exception):
    pass


@pytest.mark.parametrize('tool,method', [(ac, None), (baseline, 'gnac'),
    (baseline, 'dreamer'), (baseline, 'mbpo'), (baseline, 'ippo'),
    (baseline, 'mappo'), (baseline, 'iql'), (baseline, 'mamba'), (baseline, 'dta')])
@pytest.mark.parametrize('explicit_weight', [None, 0., 1.])
def test_training_entrypoint_default_and_explicit_off(monkeypatch, tmp_path, tool, method, explicit_weight):
    captured = []
    def resolve(config, args):
        captured.append(args)
        raise CapturedArgs
    owner = tool if tool is ac else baseline.common
    monkeypatch.setattr(owner, 'resolve_collection_settings', resolve)
    argv = ['--output-dir', str(tmp_path / 'run')]
    if method:
        argv += ['--method', method]
    if explicit_weight is not None:
        argv += ['--sp-prior-weight', str(explicit_weight)]
    with pytest.raises(CapturedArgs):
        tool.main(argv)
    args = captured[0]
    assert args.sp_prior_weight == (0. if explicit_weight is None else explicit_weight)
    assert (args.sp_prior_temperature, args.sp_prior_hold_episodes, args.sp_prior_off_episode) == (30., 200, 1000)


@pytest.mark.parametrize('explicit_weight', [None, 0., 1.])
def test_city_launch_default_and_off_are_forwarded(monkeypatch, tmp_path, explicit_weight):
    captured = []
    monkeypatch.setattr(city, 'launch', lambda args: captured.append(args))
    argv = ['launch', '--output-root', str(tmp_path / 'run')]
    if explicit_weight is not None:
        argv += ['--sp-prior-weight', str(explicit_weight)]
    city.main(argv)
    args = captured[0]
    args.cities = ['ingolstadt_custom']
    args.methods = list(city.METHODS + city.EXTRA_METHODS)
    runs = city.build_runs(tmp_path / 'run', tmp_path / 'source', args, 'test')
    expected = str(0. if explicit_weight is None else explicit_weight)
    for run in runs:
        command = run['command']
        if run['method'] == 'random':
            assert '--sp-prior-weight' not in command
        else:
            assert command[command.index('--sp-prior-weight') + 1] == expected


def test_restart_defaults_disable_prior_without_history_mask(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(restart, 'prepare', lambda args: captured.append(args))
    monkeypatch.setattr(restart.sys, 'argv', ['restart_baselines_meso.py', 'prepare',
        '--previous-root', str(tmp_path / 'old'), '--output-root', str(tmp_path / 'new')])
    restart.main()
    assert captured[0].sp_prior_weight == 0.
    assert captured[0].forbid_revisit is False


def test_legacy_missing_prior_fields_remain_zero():
    assert ac.PRIOR_DEFAULTS['sp_prior_weight'] == 0.
    assert baseline.PRIOR_DEFAULTS['sp_prior_weight'] == 0.
    baseline.validate_prior_resume({}, SimpleNamespace(**baseline.PRIOR_DEFAULTS))
    with pytest.raises(ValueError, match='sp_prior_weight'):
        baseline.validate_prior_resume({}, SimpleNamespace(**(baseline.PRIOR_DEFAULTS | {'sp_prior_weight': 1.})))
