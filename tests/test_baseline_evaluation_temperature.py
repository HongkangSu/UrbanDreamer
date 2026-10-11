"""Evaluation action temperature does not change training or random streams."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from urban_dreamer.learning.baselines import common
from urban_dreamer.models.policy import NextHopActor


def tool(name):
    spec = spec_from_file_location(name, Path(__file__).resolve().parents[1] / 'tools' / f'{name}.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('name,cls', [('ppo', 'RoutingPpoMethod'), ('dreamer', 'DreamerMethod'),
    ('mbpo', 'MbpoMethod'), ('iql', 'IndependentQMethod')])
def test_feedforward_selector_samples_tempered_distribution(name, cls, monkeypatch):
    import importlib
    module = importlib.import_module('urban_dreamer.learning.baselines.' + name)
    # Use the selector without constructing a learner or simulator.
    method = getattr(module, cls).__new__(getattr(module, cls))
    method.device = torch.device('cpu')
    method.actor = NextHopActor()
    logits = torch.tensor([[4., 0., -torch.inf]]).expand(5000, -1)
    monkeypatch.setattr(method.actor, 'forward_logits', lambda *args, **kwargs: logits)
    obs = SimpleNamespace(features=np.zeros((5000, 3, 10), dtype=np.float32),
                          mask=np.tile([True, True, False], (5000, 1)))
    torch.manual_seed(71)
    actions, metadata = method.select([], obs, None, 0., explore=False,
                                    temperature=2., sample_actions=True)
    expected = Categorical(logits=logits / 2.)
    assert np.mean(np.asarray(actions) == 1) == pytest.approx(expected.probs[0, 1].item(), abs=.015)
    assert not np.any(np.asarray(actions) == 2)
    assert metadata[0]['entropy'] == pytest.approx(expected.entropy()[0].item())
    deterministic, _ = method.select([], obs, None, 0., explore=False)
    assert deterministic == [0] * 5000


def test_graph_sampler_temperature_changes_distribution_and_rejects_invalid(monkeypatch):
    from urban_dreamer.learning.baselines.gnac import GraphNextHopActor
    actor = GraphNextHopActor(np.eye(2))
    logits = torch.tensor([[4., 0.]]).expand(5000, -1)
    monkeypatch.setattr(actor, 'forward_logits', lambda *args: logits)
    selection = actor.sample_action(None, None, None, None, None, temperature=2.)
    expected = Categorical(logits=logits / 2.)
    assert float((selection.candidate_index == 1).float().mean()) == pytest.approx(expected.probs[0, 1].item(), abs=.015)
    torch.testing.assert_close(selection.entropy, expected.entropy())
    for invalid in (True, False, 0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError, match='positive'):
            actor.sample_action(None, None, None, None, None, temperature=invalid)


def test_mamba_evaluation_samples_actions_with_deterministic_posterior(monkeypatch):
    from test_routing_mamba_baseline import make_method
    method = make_method()
    vehicles = [dict(vehicle_id=f'v{i}', departed=True) for i in range(64)]
    method.begin_episode(SimpleNamespace(vehicles=[SimpleNamespace(**v) for v in vehicles]), seed=9)
    obs = SimpleNamespace(features=np.zeros((64, method.width, 10), dtype=np.float32),
        mask=np.ones((64, method.width), dtype=bool))
    posterior_sampling = []
    original = method.world_model.observe
    def observe(*args, **kwargs):
        posterior_sampling.append(kwargs['sample'])
        return original(*args, **kwargs)
    monkeypatch.setattr(method.world_model, 'observe', observe)
    monkeypatch.setattr(method.actor, 'routing_distribution',
        lambda latent, features, mask: Categorical(logits=torch.zeros(len(latent), method.width)))
    actions, _ = method.select(vehicles, obs, None, 0., explore=False,
                               temperature=2., sample_actions=True)
    assert posterior_sampling == [False, False]
    assert len(set(actions)) > 1


@pytest.mark.parametrize('runner_name', ['train_toy_actor_critic', 'train_toy_baseline'])
def test_explicit_sampled_evaluation_uses_fixed_temperature_point2_and_restores_rng(tmp_path, runner_name):
    module = tool(runner_name)
    actor = NextHopActor()
    draws = []
    def draw():
        draws.append((random.random(), np.random.random(), torch.rand(1).item()))
    if runner_name == 'train_toy_actor_critic':
        def run(**kwargs):
            assert kwargs['policy'] == 'stochastic' and kwargs['temperature'] == .2
            draw()
            return [], dict(policy=kwargs['policy'], selection_score=-1.)
        runner = SimpleNamespace(actor=actor, run_episode=run)
        evaluate = lambda: module.evaluate(runner, tmp_path, episode=1, seeds=[41, 42], phase='evaluation', deterministic=False)
    else:
        def select(*args, **kwargs):
            assert kwargs == dict(explore=False, temperature=.2, sample_actions=True)
            draw()
        def run(select, *, seed, label, on_demand):
            select()
            return [], [], dict(policy=label, selection_score=-1.)
        method = SimpleNamespace(actor=actor, select=select)
        runner = SimpleNamespace(run_episode=run)
        evaluate = lambda: module.evaluate(runner, method, tmp_path, episode=1, seeds=[41, 42], phase='evaluation', deterministic=False)
    random.seed(32)
    np.random.seed(32)
    torch.manual_seed(32)
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    records = evaluate()
    assert {r['policy'] for r in records} == {'sampled_temperature_0.2'}
    assert all(r['evaluation_temperature'] == .2 and r['evaluation_rng_isolated'] for r in records)
    assert [r['evaluation_action_seed'] for r in records] == [41, 42]
    assert random.getstate() == states[0]
    assert np.array_equal(np.random.get_state()[1], states[1][1])
    assert torch.equal(torch.get_rng_state(), states[2])
    evaluate()
    assert draws[:2] == draws[2:]


def test_rng_restored_when_evaluation_raises():
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    generator = np.random.default_rng(11)
    generator_before = generator.bit_generator.state
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    with pytest.raises(RuntimeError):
        with common.evaluation_rng(71, SimpleNamespace(rng=generator)):
            random.random(); np.random.random(); torch.rand(1); generator.random()
            raise RuntimeError('failed')
    assert random.getstate() == states[0]
    assert np.array_equal(np.random.get_state()[1], states[1][1])
    assert torch.equal(torch.get_rng_state(), states[2])
    assert generator.bit_generator.state == generator_before


def test_cpu_evaluation_does_not_initialize_cuda_for_rng_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    def forbidden():
        raise AssertionError('CPU evaluation must not create CUDA contexts')
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', forbidden)
    with common.evaluation_rng(19):
        torch.rand(2)


@pytest.mark.parametrize('runner_name', ['train_toy_actor_critic', 'train_toy_baseline'])
def test_default_evaluation_uses_argmax(runner_name, tmp_path):
    module = tool(runner_name)
    actor = NextHopActor()
    calls = []
    if runner_name == 'train_toy_actor_critic':
        def run(**kwargs):
            calls.append(kwargs)
            return [], dict(selection_score=-1.)
        runner = SimpleNamespace(actor=actor, run_episode=run)
        records = module.evaluate(runner, tmp_path, episode=1, seeds=[71], phase='evaluation')
        assert calls == [dict(policy='deterministic', seed=71)]
    else:
        def select(*args, **kwargs):
            calls.append(kwargs)
        def run(select, *, seed, label, on_demand):
            select()
            return [], [], dict(policy=label, selection_score=-1.)
        method = SimpleNamespace(actor=actor, select=select)
        records = module.evaluate(SimpleNamespace(run_episode=run), method, tmp_path,
                                  episode=1, seeds=[71], phase='evaluation')
        assert calls == [dict(explore=False)]
    assert records[0]['policy'] == 'deterministic'
    assert records[0]['evaluation_temperature'] is None
    assert common.evaluation_policy_label() == 'deterministic'
