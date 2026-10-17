"""Real AC entropy units and an independent horizon survive stop-budget extension."""
from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from test_toy_model_free_actor_critic import args, module, rows, windows


@pytest.mark.parametrize('saved_override', [None, 1000])
def test_annealed_extension_keeps_optimizer_rng_and_episode_clock(tmp_path, saved_override):
    ac = module()
    original_args = args(tmp_path)
    original_args.episodes = 1000
    if saved_override is not None:
        original_args.entropy_schedule_episodes = saved_override
    original = ac.RealActorCritic(original_args)
    original.update(rows(), windows(original, [-.5, -2.]))
    path = tmp_path / 'latest.pt'
    original.save(path, 500, 1.25)
    saved = torch.load(path, weights_only=False)
    expected_rng = (torch.rand(4), np.random.random(4), random.random())

    extended_args = args(tmp_path)
    extended_args.episodes = 2000
    extended_args.entropy_schedule_episodes = 1000
    extended = ac.RealActorCritic(extended_args)
    assert extended.load(path) == (501, 1.25)
    torch.testing.assert_close(torch.rand(4), expected_rng[0], rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.random(4), expected_rng[1])
    assert random.random() == expected_rng[2]
    for name in ('actor_optimizer', 'global_optimizer'):
        actual = getattr(extended, name).state_dict()
        assert actual['param_groups'] == saved[name]['param_groups']
        assert actual['state'].keys() == saved[name]['state'].keys()
        for index, values in saved[name]['state'].items():
            for key, value in values.items():
                torch.testing.assert_close(actual['state'][index][key], value, rtol=0, atol=0)
    ac.set_episode_entropy(original, 501)
    assert extended.entropy_profile == original.entropy_profile
    assert extended.entropy_profile['progress'] == .5
    assert extended.entropy_profile['total_episodes'] == 1000
    original.update(rows(), windows(original, [-.5, -2.]))
    extended.update(rows(), windows(extended, [-.5, -2.]))
    for name in ('actor', 'global_critic'):
        for key, value in getattr(original, name).state_dict().items():
            torch.testing.assert_close(value, getattr(extended, name).state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize('override', [None, 999, 2000])
def test_annealed_resume_rejects_changed_effective_horizon(tmp_path, override):
    ac = module()
    original_args = args(tmp_path)
    original_args.episodes = 1000
    original = ac.RealActorCritic(original_args)
    path = tmp_path / 'latest.pt'
    original.save(path, 500, 1.)
    extended_args = args(tmp_path)
    extended_args.episodes = 2000
    extended_args.entropy_schedule_episodes = override
    extended = ac.RealActorCritic(extended_args)
    with pytest.raises(ValueError, match='entropy schedule episode budget'):
        extended.load(path)
    assert not extended.actor_optimizer.state


@pytest.mark.parametrize('mode', ['population', 'legacy', 'annealed', 'off'])
def test_update_entropy_matches_mode_population_and_event_scale(tmp_path, mode):
    ac = module()
    options = args(tmp_path)
    options.episodes = 1000
    runner = ac.RealActorCritic(options)
    runner.config = replace(runner.config,
        method=replace(runner.config.method, actor_entropy_mode=mode))
    ac.set_episode_entropy(runner, 501)
    runner.n = 37
    for parameter in runner.global_critic.parameters():
        parameter.data.zero_()
    events = rows()
    events.append({**events[0], 'features': events[0]['features'].copy()})
    macro_windows = windows(runner, [-20., -2.])
    with torch.no_grad():
        global_advantage, _, _ = runner.global_targets(macro_windows)
        expanded = global_advantage[torch.tensor([event['window'] for event in events])]
        scale = expanded.std(unbiased=False).clamp_min(1.)
        advantage = (expanded / scale).clamp(-2.5, 2.5)
        distribution = torch.distributions.Categorical(logits=runner.actor.forward_logits(
            runner.tensor(np.stack([event['features'] for event in events])),
            torch.as_tensor(np.stack([event['mask'] for event in events]))))
        coefficient = runner.entropy_coefficient
        if mode == 'population':
            coefficient = coefficient / runner.n / float(scale)
        expected = -(distribution.log_prob(torch.tensor([event['action'] for event in events]))
                     * advantage + coefficient * distribution.entropy()).mean()
    metrics = runner.update(events, macro_windows)
    assert metrics['actor_loss'] == pytest.approx(float(expected), abs=1e-6)
    assert metrics['entropy_coefficient_mean'] == pytest.approx(coefficient)
    assert metrics['entropy_advantage_scale'] == pytest.approx(float(scale))
    assert metrics['event_advantage_std'] == pytest.approx(float(expanded.std(unbiased=False)))


def test_update_diagnostics_are_detached_and_report_actual_update(tmp_path):
    ac = module()
    runner = ac.RealActorCritic(args(tmp_path))
    events = rows()
    events[0]['timeout_return'] = -5000.
    parameters = [parameter.detach().clone() for parameter in runner.actor.parameters()]
    rng = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
    with torch.no_grad():
        features = runner.tensor(np.stack([event['features'] for event in events]))
        mask = torch.as_tensor(np.stack([event['mask'] for event in events]))
        logits = runner.actor.forward_logits(features, mask)
        expected_spread = (logits.masked_fill(~mask, -torch.inf).amax(-1)
            - logits.masked_fill(~mask, torch.inf).amin(-1)).mean()
    metrics = runner.update(events, windows(runner, [-.5, -2.]))
    expected_norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(parameter.detach() - previous)
        for parameter, previous in zip(runner.actor.parameters(), parameters)]))
    assert metrics['actor_parameter_update_norm'] == pytest.approx(float(expected_norm))
    assert metrics['actor_parameter_update_norm'] > 0
    assert metrics['actor_optimizer_steps'] == 1
    assert metrics['actor_advantage_clipped_fraction'] == .5
    assert metrics['policy_logit_spread_mean'] == pytest.approx(float(expected_spread))
    assert all(not isinstance(value, torch.Tensor) for value in metrics.values())
    torch.testing.assert_close(torch.get_rng_state(), rng[0], rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
    assert np.random.get_state()[2:] == rng[1][2:]
    assert random.getstate() == rng[2]
    empty = runner.update([], windows(runner, [-.5, -2.]))
    assert empty['actor_parameter_update_norm'] == 0
    assert empty['actor_optimizer_steps'] == 1
    assert empty['actor_advantage_clipped_fraction'] == 0


@pytest.mark.parametrize('value', ['0', '-1'])
def test_cli_rejects_nonpositive_entropy_horizon(tmp_path, value):
    with pytest.raises(SystemExit):
        module().main(['--output-dir', str(tmp_path), '--entropy-schedule-episodes', value])
