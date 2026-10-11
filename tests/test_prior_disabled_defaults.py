"""New experiments must not silently re-enable the stopped batch's prior."""

import argparse
from pathlib import Path
import tomllib

import pytest

from tools import run_city_baseline_dataset, train_toy_actor_critic, train_toy_baseline


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('path', sorted((ROOT / 'configs').glob('*.toml')), ids=lambda path: path.name)
def test_shipped_mainline_configs_disable_shortest_path_prior(path):
    settings = tomllib.loads(path.read_text())
    assert settings.get('method', {}).get('sp_prior_weight', 0.) == 0.


@pytest.mark.parametrize('entrypoint', [train_toy_actor_critic, train_toy_baseline, run_city_baseline_dataset])
def test_training_cli_disables_prior_unless_explicitly_enabled(entrypoint, monkeypatch):
    class ParsedDefaults(Exception):
        pass

    def inspect_parser(parser, *args, **kwargs):
        assert parser.get_default('sp_prior_weight') == 0.
        raise ParsedDefaults

    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', inspect_parser)
    with pytest.raises(ParsedDefaults):
        entrypoint.main([])


def test_dataset_commands_without_explicit_prior_use_zero(tmp_path):
    settings = argparse.Namespace(cities=['ingolstadt_custom'],
        methods=['online_ac', 'gnac', 'dreamer', 'mbpo', 'ippo', 'mappo', 'iql', 'mamba', 'dta'],
        devices=[], episodes=5000, seed=42, checkpoint_every=1, eval_every=100, eval_seeds=3)
    runs = run_city_baseline_dataset.build_runs(tmp_path / 'batch', tmp_path / 'source', settings, 'test')
    assert len(runs) == 9
    for run in runs:
        command = run['command']
        assert float(command[command.index('--sp-prior-weight') + 1]) == 0.
