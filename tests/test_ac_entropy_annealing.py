from dataclasses import replace
from types import SimpleNamespace
import pytest
from tools.train_toy_actor_critic import set_episode_entropy
from urban_dreamer.workflow.config import MethodConfig

def runner(mode='annealed',budget=5000):
    return SimpleNamespace(config=SimpleNamespace(method=replace(MethodConfig(),actor_entropy_mode=mode)),
                           args=SimpleNamespace(episodes=budget))

def test_real_ac_resumed_episode_uses_global_budget_and_floor():
    r=runner()
    assert set_episode_entropy(r,501)['coefficient']==.002
    assert set_episode_entropy(r,2251)['coefficient']==pytest.approx(.00105)
    assert set_episode_entropy(r,4001)['coefficient']==.0001
    assert set_episode_entropy(r,5001)['coefficient']==.0001
    assert r.entropy_profile['progress']==1

def test_real_ac_historical_population_config_keeps_its_old_fixed_coefficient():
    assert set_episode_entropy(runner('population'),4001)['coefficient']==.002

def test_real_ac_off_mode_is_explicit_zero():
    assert set_episode_entropy(runner('off'),4001)['coefficient']==0
