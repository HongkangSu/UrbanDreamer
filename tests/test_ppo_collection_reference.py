"""PPO checks the collected distribution without hiding stale actor state."""
from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from test_routing_ppo_baselines import data, method
from urban_dreamer.learning.baselines.collection_policy import CollectionPolicy
from urban_dreamer.learning.baselines.ppo import IppoMethod, MappoMethod
from urban_dreamer.models.policy import set_shortest_path_prior


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def reference(model, rows):
    features, mask = model._row_tensors(rows)
    actions = torch.tensor([row['action'] for row in rows], device=model.device)
    old = torch.tensor([row['old_log_prob'] for row in rows], device=model.device)
    return model._behavior_distribution(rows, features, mask, actions, old)


def separate_collector(model):
    actor = deepcopy(model.actor)
    model.collection_policy = SimpleNamespace(device=torch.device('cpu'),
        proxy=SimpleNamespace(actor=actor))
    return actor


def assert_same_checkpoint(actual, expected):
    assert actual.keys() == expected.keys()
    assert actual['learning_protocol'] == expected['learning_protocol']
    for key in actual.keys() - {'learning_protocol'}:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_captured_failed_logits_accept_collector_reference_and_keep_strict_fallback(monkeypatch):
    # Actual Montereau EP743 row28185: same weights, CPU collection vs CUDA512.
    cpu = torch.tensor([[0., 579.35400390625, 554.0671997070312, 580.2476806640625]])
    cuda = torch.tensor([[0., 579.3539428710938, 554.067138671875, 580.2477416992188]])
    model = method(MappoMethod)
    model.collection_policy = CollectionPolicy(model, 'cpu')
    monkeypatch.setattr(model.actor, 'forward_logits', lambda features, mask: cpu.expand(len(features), -1))
    monkeypatch.setattr(model, '_all_logits', lambda features, mask: cuda.expand(len(features), -1))
    collected = float(Categorical(logits=cpu).log_prob(torch.tensor([1]))[0])
    rows = [dict(window=97, time=2910., features=np.zeros((4, 10), dtype=np.float32),
                 mask=np.ones(4, dtype=bool), action=1, old_log_prob=collected)]
    assert not torch.allclose(Categorical(logits=cuda).log_prob(torch.tensor([1])),
                             torch.tensor([collected]), atol=1e-4, rtol=1e-5)
    before = deepcopy(rows)
    actual = reference(model, rows)
    torch.testing.assert_close(actual.probs, Categorical(logits=cpu).probs, rtol=1e-6, atol=1e-7)
    assert rows[0]['old_log_prob'] == before[0]['old_log_prob']
    del model.collection_policy
    with pytest.raises(ValueError, match='behavior probabilities'):
        reference(model, rows)


def test_original_contiguous_groups_preserve_order_without_merging_repeated_keys(monkeypatch):
    model = method(MappoMethod)
    model.collection_policy = CollectionPolicy(model, 'cpu')
    source, _ = data(model)
    rows = [deepcopy(source[index]) for index in (0, 1, 2, 3, 0)]
    for i, row in enumerate(rows):
        row.update(window=[0, 0, 0, 1, 0][i], time=[0., 0., 15., 30., 0.][i])
        row['features'][0, 0] = i
    sizes = [2, 1, 1, 1]
    begin = 0
    for size in sizes:
        subset = rows[begin:begin+size]
        features, mask = model._row_tensors(subset)
        with torch.no_grad():
            distribution = Categorical(logits=model.actor.forward_logits(features, mask))
            values = distribution.log_prob(torch.tensor([row['action'] for row in subset]))
        for row, value in zip(subset, values):
            row['old_log_prob'] = float(value)
        begin += size
    calls = []
    original = model.actor.forward_logits

    def traced(features, mask):
        calls.append(features[:, 0, 0].tolist())
        return original(features, mask)

    monkeypatch.setattr(model.actor, 'forward_logits', traced)
    reference(model, rows)
    assert calls == [[0., 1.], [2.], [3.], [4.]]


@pytest.mark.parametrize('mutation', ['weight', 'buffer', 'prior', 'temperature', 'gauge', 'training'])
def test_stale_separate_collector_rejected_before_any_update(mutation):
    model = method(MappoMethod)
    if mutation == 'buffer':
        model.actor.register_buffer('audit_marker', torch.zeros(1))
    rows, windows = data(model)
    collector_actor = separate_collector(model)
    with torch.no_grad():
        if mutation == 'weight':
            collector_actor.policy_head.weight[0, 0] += 1e-6
        elif mutation == 'buffer':
            collector_actor.audit_marker.add_(1.)
        elif mutation == 'prior':
            set_shortest_path_prior(collector_actor, .5, 30.)
        elif mutation == 'temperature':
            set_shortest_path_prior(collector_actor, 0., 31.)
        elif mutation == 'gauge':
            collector_actor.score_gauge = 'pre_head_legal_reference_v1'
        else:
            collector_actor.eval()
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match='collector actor'):
        model.on_episode(rows, windows, episode=1)
    assert_same_checkpoint(model.state_dict(), before)


@pytest.mark.parametrize('mutation', ['learner_weights', 'old_log_prob'])
def test_same_device_reference_still_rejects_changed_behavior(mutation):
    model = method(MappoMethod)
    model.collection_policy = CollectionPolicy(model, 'cpu')
    rows, windows = data(model)
    if mutation == 'learner_weights':
        with torch.no_grad():
            model.actor.policy_head.weight.mul_(3.)
    else:
        rows[0]['old_log_prob'] += .01
    with pytest.raises(ValueError, match='behavior probabilities'):
        model.on_episode(rows, windows, episode=1)
    assert not model.actor_optimizer.state and not model.critic_optimizer.state


def test_normalized_logits_retain_rare_legal_action_and_illegal_mask(monkeypatch):
    model = method(MappoMethod)
    model.collection_policy = CollectionPolicy(model, 'cpu')
    monkeypatch.setattr(model.actor, 'forward_logits', lambda features, mask:
        torch.tensor([[0., -1000., -torch.inf]]).expand(len(features), -1))
    rows = [dict(window=0, time=0., features=np.zeros((3, 10), dtype=np.float32),
                 mask=np.array([True, True, False]), action=0, old_log_prob=0.)]
    actual = reference(model, rows)
    assert actual.logits[0, 1] == -1000.
    assert torch.isneginf(actual.logits[0, 2])
    assert actual.probs[0, 1] == actual.probs[0, 2] == 0.
    assert not actual.logits.requires_grad


@pytest.mark.parametrize('cls', [MappoMethod, IppoMethod])
def test_reference_preserves_rng_checkpoint_and_collected_probabilities(cls):
    model = method(cls)
    model.collection_policy = CollectionPolicy(model, 'cpu')
    rows, windows = data(model)
    observed = [row['old_log_prob'] for row in rows]
    model.on_episode(rows, windows, episode=1)
    state = deepcopy(model.state_dict())
    restored = method(cls)
    restored.load_state_dict(state)
    restored.collection_policy = CollectionPolicy(restored, 'cpu')
    next_rows, _ = data(restored)
    before = deepcopy(restored.state_dict())
    rng = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
    reference(restored, next_rows)
    assert_same_checkpoint(restored.state_dict(), before)
    torch.testing.assert_close(torch.get_rng_state(), rng[0], rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
    assert np.random.get_state()[2:] == rng[1][2:]
    assert random.getstate() == rng[2]
    assert [row['old_log_prob'] for row in rows] == observed
    assert before.keys() == state.keys()
    assert before['learning_protocol'] == state['learning_protocol']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA numeric regression')
def test_actual_failed_feature_with_cpu_collector_and_cuda_learner():
    root = Path(__file__).resolve().parents[1]
    artifact = root / 'outputs/city_baselines_current_to_dest_1000_20261013_180607/mappo_montereau_behavior_failure_20261014.json'
    if not artifact.is_file():
        pytest.skip('local recorded Montereau failing input is unavailable')
    audit = json.loads(artifact.read_text())
    checkpoint = Path(audit['checkpoint'])
    if not checkpoint.is_file():
        pytest.skip('local recorded Montereau actor checkpoint is unavailable')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['episode'] != audit['checkpoint_episode']:
        pytest.skip('recorded actor has advanced beyond the numerical audit')
    model = method(MappoMethod)
    model.device = torch.device('cuda:0')
    model.actor.to(model.device)
    model.actor.load_state_dict(state['method_state']['actor'])
    model.actor.score_gauge = audit['score_gauge']
    model.collection_policy = CollectionPolicy(model, 'cpu')
    item = next(row for row in audit['comparisons']['cuda512']['top']
                if row['absolute_error'] > row['threshold'])
    features = np.asarray(item['features'], dtype=np.float32)
    mask = np.asarray(item['mask'], dtype=bool)
    with torch.no_grad():
        actor = model.collection_policy.proxy.actor
        distribution = Categorical(logits=actor.forward_logits(
            torch.from_numpy(features[None]).expand(512, -1, -1),
            torch.from_numpy(mask[None]).expand(512, -1)))
        old = float(distribution.log_prob(torch.full((512,), item['selected_slot']))[0])
    rows = [dict(window=item['window'], time=30.*item['window'], features=features,
                 mask=mask, action=item['selected_slot'], old_log_prob=old)] * 512
    actual = reference(model, rows)
    assert actual.logits.device == model.device
    torch.testing.assert_close(actual.probs.cpu(), distribution.probs, rtol=1e-5, atol=1e-6)
    assert all(row['old_log_prob'] == old for row in rows)
