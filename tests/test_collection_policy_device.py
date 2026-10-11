from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import train_toy_baseline as trainer
from urban_dreamer.learning.baselines.collection_policy import ActorMirror, CollectionPolicy
from urban_dreamer.learning.baselines.ppo import MappoMethod
from urban_dreamer.models.policy import NextHopActor, set_shortest_path_prior


def method(device='cpu'):
    torch.set_num_threads(1)
    args = SimpleNamespace(seed=42, actor_lr=2e-4, critic_lr=3e-4)
    runner = SimpleNamespace(device=torch.device(device), network=SimpleNamespace(edge_count=3))
    return MappoMethod(args, runner)


def test_mirror_is_frozen_until_next_sync_and_copies_prior():
    actor = NextHopActor()
    mirror = ActorMirror('cuda:0', 'cpu')  # Test copying on CPU even without CUDA hardware.
    set_shortest_path_prior(actor, .8, 17.)
    copy = mirror.sync(actor)
    before = [p.clone() for p in copy.parameters()]
    with torch.no_grad():
        for p in actor.parameters(): p.add_(.125)
    set_shortest_path_prior(actor, 0., 30.)
    assert all(torch.equal(p, old) for p, old in zip(copy.parameters(), before))
    assert copy.prior_weight == .8
    assert all(p.requires_grad for p in actor.parameters())
    assert not any(p.requires_grad for p in copy.parameters())
    assert mirror.sync(actor) is copy
    assert copy.prior_weight == 0. and copy.prior_temperature_seconds == 30.
    for p, original in zip(copy.parameters(), actor.parameters()): torch.testing.assert_close(p, original)


def test_same_device_does_not_copy_or_change_gradient_flags():
    actor = NextHopActor()
    assert ActorMirror('cpu').sync(actor) is actor
    assert all(p.requires_grad for p in actor.parameters())


def test_stateful_cross_device_copy_is_rejected():
    with pytest.raises(ValueError, match='feed-forward'):
        ActorMirror('cuda:0', 'cpu').sync(torch.nn.Linear(3, 2))


def test_eval_prior_and_updated_weights_sync_before_every_seed(tmp_path):
    learner = method()
    learner.device = torch.device('cuda:0')  # CPU-only test of the migration path.
    learner.collection_policy = CollectionPolicy(learner, 'cpu')
    collector = learner.collection_policy
    set_shortest_path_prior(learner.actor, .6, 30.)
    seen = []

    def run(select, *, seed, label, on_demand):
        on_demand(SimpleNamespace(vehicles=[]), seed=seed)
        seen.append((seed, label, collector.proxy.actor.prior_weight))
        return [], [], dict(policy=label, selection_score=0.)

    trainer.evaluate(SimpleNamespace(run_episode=run), learner, tmp_path, episode=1,
        seeds=[10042, 10043], phase='evaluation', policies=['deterministic', 'deterministic_no_prior'])
    assert [r[2] for r in seen] == [.6, .6, 0., 0.]
    assert learner.actor.prior_weight == .6
    saved = learner.state_dict()
    assert 'collection_policy' not in saved
    with torch.no_grad():
        for p in learner.actor.parameters(): p.add_(.1)
    collector.begin_episode(SimpleNamespace(vehicles=[]), seed=44)
    assert collector.proxy.actor.prior_weight == .6
    for p, q in zip(learner.actor.parameters(), collector.proxy.actor.parameters()):
        torch.testing.assert_close(p, q)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_real_cuda_learner_cpu_collection_probabilities_and_behavior_log_prob():
    learner = method('cuda:0')
    set_shortest_path_prior(learner.actor, 1., 30.)
    collector = CollectionPolicy(learner, 'cpu')
    collector.begin_episode(SimpleNamespace(vehicles=[]), seed=43)
    features = np.random.default_rng(42).uniform(0., .25, (4, 3, 10)).astype(np.float32)
    mask = np.array([[True, True, False]] * 4)
    observations = SimpleNamespace(features=features, mask=mask)
    indices, aux = collector.select([{}] * 4, observations, None, 0.)
    with torch.no_grad():
        gpu = learner.actor.forward_logits(torch.as_tensor(features, device='cuda:0'),
                                           torch.as_tensor(mask, device='cuda:0')).cpu()
        cpu = collector.proxy.actor.forward_logits(torch.from_numpy(features), torch.from_numpy(mask))
    torch.testing.assert_close(gpu.softmax(-1), cpu.softmax(-1), rtol=1e-5, atol=1e-6)
    log_prob = torch.distributions.Categorical(logits=gpu).log_prob(torch.tensor(indices))
    np.testing.assert_allclose([a['old_log_prob'] for a in aux], log_prob.numpy(), rtol=1e-5, atol=1e-6)
    assert next(learner.actor.parameters()).device.type == 'cuda'
    assert next(collector.proxy.actor.parameters()).device.type == 'cpu'


def test_checkpoint_host_loading_keeps_replay_off_gpu(tmp_path):
    from urban_dreamer.learning.baselines import common
    path = tmp_path / 'checkpoint.pt'
    prepared = SimpleNamespace(input_hashes={})
    config = SimpleNamespace(semantic_fingerprint='fixed')
    common.save_checkpoint(path, method_name='test', episode=1, best=0.,
        method_state={'replay_tensor': torch.ones(4)}, args=SimpleNamespace(),
        prepared=prepared, config=config)
    state = common.load_checkpoint(path, method_name='test', prepared=prepared,
                                   config=config, device='cuda:0')
    assert state['method_state']['replay_tensor'].device.type == 'cpu'
