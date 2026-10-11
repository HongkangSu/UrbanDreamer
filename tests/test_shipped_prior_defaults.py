"""User-facing mainline presets disable the optional behavior prior explicitly."""
from pathlib import Path

import pytest

from urban_dreamer.workflow.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


@pytest.mark.parametrize('path', sorted(CONFIGS.glob('*.toml')), ids=lambda p: p.stem)
def test_mainline_presets_disable_shared_prior(path):
    method = load_config(path).method
    assert (method.sp_prior_weight, method.sp_prior_temperature,
            method.sp_prior_hold_episodes, method.sp_prior_off_episode) == (0., 30., 200, 1000)
