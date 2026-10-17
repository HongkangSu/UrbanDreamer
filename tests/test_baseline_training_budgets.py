"""Speed and stability budgets reach training before any experiment is created."""
import pytest

from tools import train_toy_baseline as baseline


class Captured(Exception):
    pass


def test_model_based_budget_options_reach_runner(monkeypatch, tmp_path):
    seen = []

    def capture(config, args):
        seen.append(args)
        raise Captured

    monkeypatch.setattr(baseline, 'load_config', lambda path: object())
    monkeypatch.setattr(baseline.common, 'resolve_collection_settings', capture)
    options = dict(mbpo_model_train_rounds=1, mbpo_rollout_seeds=2, mbpo_rollout_length=3,
        dreamer_world_model_updates=2, dreamer_imagination_passes=3,
        dreamer_imagination_rollouts=4, dreamer_imagined_horizon=5, mamba_target_kl=.02,
        ppo_max_policy_kl=.08, ppo_max_backtracks=2)
    argv = ['--method', 'dreamer', '--output-dir', str(tmp_path / 'run')]
    for name, value in options.items():
        argv.extend(['--' + name.replace('_', '-'), str(value)])
    with pytest.raises(Captured):
        baseline.main(argv)
    assert all(getattr(seen[0], key) == value for key, value in options.items())
    assert seen[0].score_gauge == 'pre_head_legal_reference_v1'
    assert not (tmp_path / 'run').exists()


@pytest.mark.parametrize('name,value', [
    ('mbpo-model-train-rounds', '0'), ('mbpo-rollout-seeds', '-1'), ('mbpo-rollout-length', '0'),
    ('dreamer-world-model-updates', '0'), ('dreamer-imagination-passes', '0'),
    ('dreamer-imagination-rollouts', '0'), ('dreamer-imagined-horizon', '-1'),
    ('mamba-target-kl', '-1'), ('mamba-target-kl', 'nan'), ('mamba-target-kl', 'inf'),
    ('ppo-target-kl', 'nan'), ('ppo-max-policy-kl', '-1'),
    ('ppo-max-policy-kl', 'inf'), ('ppo-max-backtracks', '-1'),
])
def test_invalid_budget_rejected_before_config_or_output(monkeypatch, tmp_path, name, value):
    def unexpected(path):
        pytest.fail('invalid budget must fail before config loading')

    monkeypatch.setattr(baseline, 'load_config', unexpected)
    with pytest.raises(SystemExit) as error:
        baseline.main(['--method', 'dreamer', '--output-dir', str(tmp_path / 'run'),
                       '--' + name, value])
    assert error.value.code == 2
    assert not (tmp_path / 'run').exists()
