"""Opt-in policy score gauging preserves distributions without large offsets."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.distributions import Categorical

from urban_dreamer.learning.baselines.collection_policy import ActorMirror
from urban_dreamer.learning.baselines.ppo import IppoMethod, MappoMethod
from urban_dreamer.models.policy import NextHopActor, apply_shortest_path_prior, set_shortest_path_prior

GAUGE = 'pre_head_legal_reference_v1'


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def actor_inputs(dtype=torch.float64):
    generator = torch.Generator().manual_seed(173)
    features = torch.rand((6, 5, 10), generator=generator, dtype=dtype)
    mask = torch.tensor([[False, True, True, False, True], [True, False, True, False, False],
                         [False, False, False, True, True], [False, True, False, False, False],
                         [True, True, True, True, True], [False, False, True, True, False]])
    return features, mask


def actors(dtype=torch.float64):
    torch.manual_seed(191)
    legacy = NextHopActor().to(dtype=dtype)
    stable = deepcopy(legacy)
    stable.score_gauge = GAUGE
    return legacy, stable


def distribution(actor, features, mask, temperature=1., costs=None):
    return Categorical(logits=actor.forward_logits(features, mask,
                       prior_costs_seconds=costs) / temperature)


def test_legacy_default_is_bit_identical_and_checkpoint_layout_is_unchanged():
    legacy, stable = actors(torch.float32)
    features, mask = actor_inputs(torch.float32)
    expected = legacy.policy_head(legacy.candidate_scorer(features)).squeeze(-1).masked_fill(~mask, -torch.inf)
    expected = apply_shortest_path_prior(expected, features, mask, weight=0., temperature_seconds=30.)
    assert torch.equal(legacy.forward_logits(features, mask), expected)
    assert legacy.state_dict().keys() == stable.state_dict().keys()
    assert 'policy_head.bias' in stable.state_dict()
    stable.load_state_dict(legacy.state_dict(), strict=True)


@pytest.mark.parametrize('temperature', [1., .2, 2.])
@pytest.mark.parametrize('explicit_costs', [False, True])
def test_float64_distribution_and_gradients_match_legacy_with_prior(temperature, explicit_costs):
    legacy, stable = actors()
    features, mask = actor_inputs()
    for actor in (legacy, stable):
        set_shortest_path_prior(actor, .7, 30.)
    costs = features[..., 4] * 200. if explicit_costs else None
    old = distribution(legacy, features, mask, temperature, costs)
    new = distribution(stable, features, mask, temperature, costs)
    torch.testing.assert_close(new.probs, old.probs, atol=1e-13, rtol=1e-12)
    actions = mask.long().argmax(-1)
    weights = torch.arange(1, 7, dtype=features.dtype)
    (-old.log_prob(actions) * weights).mean().backward()
    (-new.log_prob(actions) * weights).mean().backward()
    for (name, parameter), (other_name, other) in zip(legacy.named_parameters(), stable.named_parameters()):
        assert name == other_name
        if name == 'policy_head.bias':
            assert other.grad is None
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad), atol=1e-13, rtol=0.)
        else:
            torch.testing.assert_close(other.grad, parameter.grad, atol=1e-13, rtol=1e-11)


def test_scalar_head_bias_cannot_change_stable_scores_or_probabilities():
    _, stable = actors(torch.float32)
    features, mask = actor_inputs(torch.float32)
    before = stable.forward_logits(features, mask)
    with torch.no_grad():
        stable.policy_head.bias.fill_(2. ** 24)
    after = stable.forward_logits(features, mask)
    assert torch.equal(before, after)
    assert torch.equal(Categorical(logits=before).probs, Categorical(logits=after).probs)


def test_reference_is_legal_and_illegal_padding_cannot_change_legal_scores():
    _, stable = actors(torch.float32)
    features, mask = actor_inputs(torch.float32)
    before = stable.forward_logits(features, mask)
    padded = features.clone()
    padded[~mask] = 100000.
    after = stable.forward_logits(padded, mask)
    assert torch.equal(before, after)
    assert torch.isneginf(after[~mask]).all()
    first = mask.long().argmax(-1)
    assert torch.equal(after[torch.arange(len(mask)), first], torch.zeros(len(mask)))


def test_candidate_permutation_and_batch_partition_preserve_probabilities():
    _, stable = actors()
    features, mask = actor_inputs()
    before = distribution(stable, features, mask).probs
    permutation = torch.tensor([4, 3, 2, 1, 0])
    permuted = distribution(stable, features[:, permutation], mask[:, permutation]).probs
    torch.testing.assert_close(permuted[:, permutation], before, atol=1e-13, rtol=1e-12)
    partitioned = torch.cat([distribution(stable, features[i:i+1], mask[i:i+1]).probs
                             for i in range(len(features))])
    torch.testing.assert_close(partitioned, before, atol=1e-13, rtol=1e-12)


def test_zero_event_batch_and_single_legal_candidate():
    _, stable = actors(torch.float32)
    assert stable.forward_logits(torch.empty(0, 2, 10), torch.empty(0, 2, dtype=torch.bool)).shape == (0, 2)
    features, mask = actor_inputs(torch.float32)
    probabilities = distribution(stable, features[3:4], mask[3:4]).probs
    assert torch.equal(probabilities, mask[3:4].float())


def test_adam_state_restores_exactly_and_continues_after_explicit_gauge_change():
    legacy, stable = actors()
    features, mask = actor_inputs()
    actions = mask.long().argmax(-1)
    optimizer = torch.optim.Adam(legacy.parameters(), lr=2e-4)
    (-distribution(legacy, features, mask).log_prob(actions).mean()).backward()
    optimizer.step()
    actor_state = deepcopy(legacy.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    stable.load_state_dict(actor_state, strict=True)
    resumed_optimizer = torch.optim.Adam(stable.parameters(), lr=2e-4)
    resumed_optimizer.load_state_dict(optimizer_state)
    for parameter, old in zip(stable.parameters(), legacy.parameters()):
        assert torch.equal(parameter, old)
    for restored, saved in zip(resumed_optimizer.state.values(), optimizer_state['state'].values()):
        assert restored.keys() == saved.keys()
        for key in restored:
            assert torch.equal(restored[key], saved[key])
    before = deepcopy(stable.state_dict())
    resumed_optimizer.zero_grad(set_to_none=True)
    loss = -distribution(stable, features, mask).log_prob(actions).mean()
    loss.backward()
    resumed_optimizer.step()
    assert stable.score_gauge == GAUGE
    assert torch.equal(stable.policy_head.bias, before['policy_head.bias'])
    assert not torch.equal(stable.policy_head.weight, before['policy_head.weight'])
    assert torch.isfinite(loss)


@pytest.mark.parametrize('cls', [MappoMethod, IppoMethod])
def test_ppo_opt_in_and_real_stale_actor_rejection_before_any_update(cls):
    args = SimpleNamespace(reward_objective='legacy_exposure', seed=42, actor_lr=3e-4,
                           critic_lr=3e-4, score_gauge=GAUGE)
    runner = SimpleNamespace(device=torch.device('cpu'), network=SimpleNamespace(edge_count=2), deadline=30.)
    method = cls(args, runner)
    features, mask = actor_inputs(torch.float32)
    indices, aux = method.select([{}] * 6, SimpleNamespace(features=features.numpy(), mask=mask.numpy()), None, 0.)
    rows = [dict(vehicle_id=str(i), window=0, time=0., features=features[i].numpy(),
                 mask=mask[i].numpy(), action=indices[i], **aux[i]) for i in range(6)]
    windows = [dict(obs=torch.zeros(10).numpy(), next_obs=torch.zeros(10).numpy(), reward=-1.,
                    terminal=True, start_time=0., end_time=30.)]
    critic_before = deepcopy(method.critic.state_dict())
    with torch.no_grad():
        method.actor.policy_head.weight.mul_(1000.)
    with pytest.raises(ValueError, match='do not match the frozen actor'):
        method.on_episode(rows, windows, episode=1)
    for name, parameter in method.critic.state_dict().items():
        assert torch.equal(parameter, critic_before[name])
    assert not method.actor_optimizer.state and not method.critic_optimizer.state


def test_actor_mirror_preserves_explicit_gauge_through_repeated_sync():
    _, stable = actors(torch.float32)
    mirror = ActorMirror('cuda:0', 'cpu')
    copy = mirror.sync(stable)
    assert copy.score_gauge == GAUGE
    with torch.no_grad():
        stable.policy_head.weight.add_(.25)
    mirror.sync(stable)
    features, mask = actor_inputs(torch.float32)
    assert torch.equal(copy.forward_logits(features, mask), stable.forward_logits(features, mask))


def test_unsupported_gauge_is_rejected():
    _, stable = actors()
    stable.score_gauge = 'unknown'
    features, mask = actor_inputs()
    with pytest.raises(ValueError, match='unsupported actor score_gauge'):
        stable.forward_logits(features, mask)


@pytest.mark.parametrize('cls', [MappoMethod, IppoMethod])
def test_gauge_requires_explicit_migration_under_stable_ppo_protocol(cls):
    args = SimpleNamespace(seed=42, actor_lr=3e-4, critic_lr=3e-4)
    runner = SimpleNamespace(device=torch.device('cpu'), network=SimpleNamespace(edge_count=2))
    legacy = cls(args, runner)
    explicit_legacy = cls(SimpleNamespace(**vars(args), score_gauge='legacy'), runner)
    stable = cls(SimpleNamespace(**vars(args), score_gauge=GAUGE), runner)
    assert legacy.learning_protocol() == explicit_legacy.learning_protocol()
    assert legacy.learning_protocol()['version'] == 7
    assert 'score_gauge' not in legacy.learning_protocol()
    assert stable.learning_protocol()['version'] == 7
    assert stable.learning_protocol()['score_gauge'] == GAUGE
    state = deepcopy(legacy.state_dict())
    before = deepcopy(stable.actor.state_dict())
    with pytest.raises(ValueError, match='explicitly migrate'):
        stable.load_state_dict(state)
    for name, value in stable.actor.state_dict().items():
        assert torch.equal(value, before[name])
    migrated = deepcopy(state)
    migrated['learning_protocol'] = stable.learning_protocol()
    stable.load_state_dict(migrated)
    with pytest.raises(ValueError, match='explicitly migrate'):
        legacy.load_state_dict(stable.state_dict())


@pytest.mark.parametrize('gauge', ['legacy', GAUGE])
def test_cli_parses_gauge_and_matches_ppo_contract(monkeypatch, tmp_path, gauge):
    from tools import train_toy_baseline as trainer
    captured = {}
    original = trainer.argparse.ArgumentParser.parse_args

    class Parsed(Exception):
        pass

    def capture(parser, argv):
        parsed = original(parser, argv)
        captured.update(vars(parsed))
        raise Parsed

    monkeypatch.setattr(trainer.argparse.ArgumentParser, 'parse_args', capture)
    command = ['--method', 'mappo', '--output-dir', str(tmp_path)]
    if gauge != 'legacy':
        command += ['--score-gauge', gauge]
    with pytest.raises(Parsed):
        trainer.main(command)
    assert captured['score_gauge'] == gauge
    runner = SimpleNamespace(device=torch.device('cpu'), network=SimpleNamespace(edge_count=2))
    method = MappoMethod(SimpleNamespace(**captured), runner)
    assert method.actor.score_gauge == gauge
    assert method.learning_protocol().get('score_gauge', 'legacy') == gauge


def test_cli_rejects_non_ppo_gauge_before_any_setup(tmp_path, capsys):
    from tools import train_toy_baseline as trainer
    output = tmp_path / 'not_created'
    with pytest.raises(SystemExit) as error:
        trainer.main(['--method', 'iql', '--output-dir', str(output), '--score-gauge', GAUGE])
    assert error.value.code == 2
    assert 'requires mappo or ippo' in capsys.readouterr().err
    assert not output.exists()
