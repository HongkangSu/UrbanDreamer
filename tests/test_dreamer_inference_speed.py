"""Inference acceleration preserves the sampled policy and learner isolation."""
import copy

import numpy as np
import pytest
import torch

from urban_dreamer.learning.baselines import common
from urban_dreamer.learning.baselines.dreamer import DreamerMethod
from urban_dreamer.models.policy import set_shortest_path_prior
from test_toy_baselines_model_based import make_args, active_snapshot, synthetic_episode


def test_cpu_imagination_actor_is_learner_and_keeps_rng(tmp_path):
    runner = common.EpisodeRunner(make_args(tmp_path))
    method = DreamerMethod(runner.args, runner)
    actor = method._imagination_actor()
    assert actor is method.actor
    features = torch.rand(8, 3, 10)
    mask = torch.ones(8, 3, dtype=torch.bool)
    torch.manual_seed(917)
    expected = actor.sample_action(features, mask).candidate_index
    expected_rng = torch.get_rng_state()
    torch.manual_seed(917)
    actual = method._sample_imagined_actions(actor, features.numpy(), mask.numpy())
    assert actual == expected.tolist()
    assert torch.equal(expected_rng, torch.get_rng_state())


def test_checkpoint_records_imagination_device_and_rejects_silent_change(tmp_path):
    runner = common.EpisodeRunner(make_args(tmp_path))
    method = DreamerMethod(runner.args, runner)
    state = copy.deepcopy(method.state_dict())
    assert state['imagination_policy']['device'] == 'cpu'
    state['imagination_policy']['device'] = 'cuda:0'
    with pytest.raises(ValueError, match='imagination'):
        method.load_state_dict(state)
    # A legacy CPU actor on a CPU learner requires no migration permission.
    state.pop('imagination_policy')
    method.load_state_dict(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_legacy_checkpoint_requires_explicit_device_migration(tmp_path):
    runner = common.EpisodeRunner(make_args(tmp_path, device='cuda:0', dreamer_imagination_device='auto'))
    method = DreamerMethod(runner.args, runner)
    state = copy.deepcopy(method.state_dict())
    state.pop('imagination_policy')
    with pytest.raises(ValueError, match='legacy.*imagination'):
        method.load_state_dict(state)
    method.args.dreamer_allow_legacy_imagination_migration = True
    method.load_state_dict(state)
    assert method.learning_protocol()['legacy_imagination_migration_allowed'] is True
    assert method.state_dict()['imagination_policy']['device'] == 'cpu'
    # Opt-in only handles unmarked legacy states, not new marked device changes.
    marked = copy.deepcopy(method.state_dict())
    marked['imagination_policy']['device'] = 'cuda:0'
    with pytest.raises(ValueError, match='changed across checkpoint'):
        method.load_state_dict(marked)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cpu_imagination_mirror_resyncs_weights_prior_without_moving_learner(tmp_path):
    runner = common.EpisodeRunner(make_args(tmp_path, device='cuda:0', dreamer_imagination_device='cpu'))
    method = DreamerMethod(runner.args, runner)
    set_shortest_path_prior(method.actor, .75, 17.)
    mirror = method._imagination_actor()
    assert next(mirror.parameters()).device.type == 'cpu'
    assert next(method.actor.parameters()).device.type == 'cuda'
    assert mirror.prior_weight == .75
    features = torch.rand(16, 3, 10)
    mask = torch.ones(16, 3, dtype=torch.bool)
    torch.testing.assert_close(mirror.forward_logits(features, mask),
        method.actor.forward_logits(features.cuda(), mask.cuda()).cpu(), rtol=1e-5, atol=1e-5)
    with torch.no_grad():
        method.actor.policy_head.bias.add_(2.)
    set_shortest_path_prior(method.actor, .2, 30.)
    assert method._imagination_actor() is mirror
    assert mirror.prior_weight == .2
    for source, target in zip(method.actor.parameters(), mirror.parameters()):
        torch.testing.assert_close(source.cpu(), target, rtol=0, atol=0)
    assert not any(p.requires_grad for p in mirror.parameters())


def test_standalone_rollout_matches_previous_sampling_path_on_cpu(tmp_path):
    runner = common.EpisodeRunner(make_args(tmp_path))
    method = DreamerMethod(runner.args, runner)
    method.seed_buffer.append_episode(synthetic_episode(runner, np.random.default_rng(2), steps=1,
        vehicles=(active_snapshot(runner),)), total_vehicle_count=runner.n, total_travel_minutes=1.)
    rng = copy.deepcopy(method.rng.bit_generator.state)
    optimized_sampler = method._sample_imagined_actions
    def previous_sampler(actor, features, mask):
        with torch.no_grad():
            return actor.sample_action(torch.as_tensor(features, dtype=torch.float32),
                torch.as_tensor(mask)).candidate_index.tolist()
    method._sample_imagined_actions = previous_sampler
    torch.manual_seed(813)
    first = method._imagine_rollout(0)
    old_torch_rng = torch.get_rng_state()
    method._sample_imagined_actions = optimized_sampler
    method.rng.bit_generator.state = rng
    torch.manual_seed(813)
    second = method._imagine_rollout(0)
    assert torch.equal(old_torch_rng, torch.get_rng_state())
    assert [r['action'] for r in first[0]] == [r['action'] for r in second[0]]
    assert first[1][2:] == second[1][2:]
    for left, right in zip(first[1][0], second[1][0]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
