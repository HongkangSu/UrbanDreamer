from copy import deepcopy
from collections import OrderedDict

import pytest
import torch

from tools.continue_city_paper_reward import migrate_state, preserved_learning_digest


def checkpoint():
    actor = {'weight': torch.arange(6.).reshape(2, 3)}
    critic = {'weight': torch.arange(10.).reshape(2, 5)}
    optimizer = {'state': {0: {'step': torch.tensor(12.),
        'exp_avg': torch.ones(2, 5), 'exp_avg_sq': torch.full((2, 5), 2.)}},
        'param_groups': [{'params': [0], 'lr': .001}]}
    state = dict(episode=31, best=4., config_fingerprint='old', args=dict(seed=42, global_reward_only=False),
        torch_rng=torch.arange(10), method_state=dict(actor=actor, critic=critic,
        critic_optimizer=optimizer, replay=[{'reward': -5000.}], updates=12,
        learning_protocol={'reward': 'old'}))
    template = dict(shapes={'actor': {'weight': [2, 3]}, 'critic': {'weight': [2, 7]}},
        optimizer_parameters={'critic_optimizer': [('critic', 'weight')]},
        metadata={'learning_protocol': {'reward': 'observed_mean'}}, empty_buffers={'replay': []})
    return state, template


def test_migration_preserves_learning_state_and_zero_extends_adam():
    state, template = checkpoint()
    original = deepcopy(state)
    digest = preserved_learning_digest(state, template)
    result = migrate_state(state, template)
    assert preserved_learning_digest(state, template, result['expansions']) == digest
    assert state['episode'] == 31
    assert state['args']['global_reward_only'] is True
    assert state['args']['reward_objective'] == 'observed_mean'
    assert state['best'] == float('inf')
    method = state['method_state']
    assert method['replay'] == []
    assert method['updates'] == 12
    assert torch.equal(method['actor']['weight'], original['method_state']['actor']['weight'])
    assert torch.equal(method['critic']['weight'][:, :5], original['method_state']['critic']['weight'])
    assert torch.count_nonzero(method['critic']['weight'][:, 5:]) == 0
    moments = method['critic_optimizer']['state'][0]
    assert moments['step'] == 12
    assert torch.equal(moments['exp_avg'][:, :5], torch.ones(2, 5))
    assert torch.count_nonzero(moments['exp_avg'][:, 5:]) == 0


def test_migration_refuses_shrinking_context_or_actor_expansion():
    state, template = checkpoint()
    template['shapes']['critic']['weight'] = [2, 4]
    with pytest.raises(ValueError, match='shrink'):
        migrate_state(state, template)
    state, template = checkpoint()
    template['shapes']['actor']['weight'] = [2, 5]
    with pytest.raises(ValueError, match='actor'):
        migrate_state(state, template)


def test_preservation_proof_detects_unrelated_learning_change():
    state, template = checkpoint()
    digest = preserved_learning_digest(state, template)
    result = migrate_state(state, template)
    state['method_state']['actor']['weight'][0, 0] += 1
    assert preserved_learning_digest(state, template, result['expansions']) != digest


def test_preservation_proof_handles_pytorch_state_dict_metadata():
    state, template = checkpoint()
    original = OrderedDict(state['method_state']['critic'])
    original._metadata = OrderedDict({'': {'version': 1}})
    state['method_state']['critic'] = original
    digest = preserved_learning_digest(state, template)
    result = migrate_state(state, template)
    assert preserved_learning_digest(state, template, result['expansions']) == digest
