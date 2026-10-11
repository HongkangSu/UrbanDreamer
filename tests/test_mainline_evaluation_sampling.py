"""Sampled evaluation remains repeatable and cannot perturb training RNG streams."""
import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.run_online_pilot import OnlinePilot
from urban_dreamer.learning.training import MainlineTrainer, evaluation_protocol
from test_online_pilot import FakeTrainer
from test_urban_dreamer_training import make_trainer, fixed_demand, _complete_demand_session


@pytest.mark.parametrize('temperature', [0., -1., float('inf'), float('nan')])
def test_invalid_evaluation_temperature_rejected(temperature):
    with pytest.raises(ValueError, match='temperature'):
        evaluation_protocol(temperature)


def test_default_protocol_records_argmax_guidance_and_fixed_action_seed():
    protocol = evaluation_protocol(demand_seed=42)
    assert protocol['policy'] == 'argmax'
    assert protocol['archive_policy'] == 'zero_prior_argmax'
    assert protocol['action_seed_base'] == 1000045
    assert protocol['temperature'] is None
    assert protocol['training_rng_isolated']
    assert evaluation_protocol(None, prior_weight=.5)['archive_policy'] == 'shortest_path_prior_argmax'
    assert evaluation_protocol(.2)['policy'] == 'sampled_temperature_0.2'


@pytest.mark.parametrize('raise_error', [False, True])
def test_evaluation_repeats_action_randomness_and_restores_training_rng(raise_error):
    trainer = SimpleNamespace(config=SimpleNamespace(method=SimpleNamespace(evaluation_demand_seed=42)),
                              rng=np.random.default_rng(17), evaluation_temperature=2.)
    draws = []
    def collect(**kwargs):
        draws.append((random.random(), np.random.random(), trainer.rng.random(), torch.rand(3)))
        if raise_error:
            raise RuntimeError('interrupted evaluation')
        return draws[-1]
    trainer._collect_episode = collect
    random.seed(13)
    np.random.seed(14)
    torch.manual_seed(15)
    original_generator = trainer.rng
    python_state, numpy_state = random.getstate(), np.random.get_state()
    generator_state = copy.deepcopy(trainer.rng.bit_generator.state)
    torch_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    for _ in range(2):
        if raise_error:
            with pytest.raises(RuntimeError, match='interrupted'):
                MainlineTrainer._run_episode(trainer, mode='evaluation', episode_index=1)
        else:
            MainlineTrainer._run_episode(trainer, mode='evaluation', episode_index=1)
        assert random.getstate() == python_state
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        assert trainer.rng is original_generator
        assert trainer.rng.bit_generator.state == generator_state
        assert torch.equal(torch.get_rng_state(), torch_state)
        if cuda_state:
            assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_state))
    assert draws[0][:3] == draws[1][:3]
    torch.testing.assert_close(draws[0][3], draws[1][3])
    if not raise_error:
        MainlineTrainer._run_episode(trainer, mode='evaluation', episode_index=2)
        assert draws[2][:3] != draws[0][:3]


@pytest.mark.parametrize('mode,temperature,expected', [
    ('online', 2., (False, 1.)), ('evaluation', 2., (False, 2.)),
    ('evaluation', None, (True, 1.))])
def test_action_selection_changes_only_evaluation_temperature(tmp_path, monkeypatch, mode, temperature, expected):
    trainer = make_trainer(tmp_path)
    trainer.evaluation_temperature = temperature
    calls = []
    def select(features, mask, **kwargs):
        calls.append((kwargs['deterministic'], kwargs['temperature']))
        return SimpleNamespace(candidate_index=torch.tensor([0]))
    monkeypatch.setattr(trainer.actor, 'sample_action', select)
    states = [dict(vehicle_id='v', current_edge=0)]
    observations = SimpleNamespace(features=np.zeros((1, 1, 10), dtype=np.float32),
                                   mask=np.ones((1, 1), dtype=bool), edge_indices=np.array([[1]]))
    trainer._select(states, None, 0., context=None, mode=mode, observations=observations)
    assert calls == [expected]


def test_archive_records_actual_default_argmax_evaluation_protocol(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.demand = fixed_demand(trainer, lambda **kwargs: trainer.topology.vehicles[:1])
    trainer._session_factory = _complete_demand_session(trainer, [])
    trainer._run_episode(mode='evaluation', episode_index=0)
    with np.load(tmp_path / 'episodes' / 'evaluation' / 'episode_000000.npz', allow_pickle=False) as archive:
        metadata = json.loads(archive['summary_json'].tobytes().decode())['metadata']
    assert metadata['evaluation_policy'] == 'zero_prior_argmax'
    assert metadata['evaluation_temperature'] is None
    assert metadata['evaluation_action_seed'] == trainer.config.method.evaluation_demand_seed + 1000003


def test_resume_rejects_changed_evaluation_policy_before_training(tmp_path):
    trainer = FakeTrainer(tmp_path)
    base = dict(source=dict(sha256='frozen'), evaluation=evaluation_protocol(None))
    runner = OnlinePilot(trainer, output=trainer.config.run.output_dir, input_hash='inputs', protocol=base)
    runner.save()
    output = trainer.config.run.output_dir
    (output / 'pilot_protocol.json').write_text(json.dumps(runner.protocol))
    resumed = OnlinePilot(trainer, output=output, input_hash='inputs',
        protocol=dict(base, evaluation=evaluation_protocol(.2)))
    with pytest.raises(ValueError, match='protocol/source identity'):
        resumed.resume()
    assert not trainer.train_calls
