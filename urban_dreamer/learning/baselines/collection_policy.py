"""Episode-frozen inference copies; learner parameters and optimizers stay put."""
from __future__ import annotations

import copy

import torch

from ...models.policy import NextHopActor, set_shortest_path_prior


class ActorMirror:
    def __init__(self, learner_device, collection_device=None):
        self.learner_device = torch.device(learner_device)
        self.device = torch.device(collection_device or learner_device)
        self.actor = None

    def sync(self, actor):
        if self.device == self.learner_device:
            return actor
        if not isinstance(actor, NextHopActor):
            raise ValueError('separate collection device requires a feed-forward NextHopActor; '
                             'stateful/graph actors must retain their learner device')
        with torch.no_grad():
            if self.actor is None:
                self.actor = copy.deepcopy(actor).to(self.device).requires_grad_(False)
            else:
                self.actor.load_state_dict(actor.state_dict())
            self.actor.train(actor.training)
            set_shortest_path_prior(self.actor, actor.prior_weight, actor.prior_temperature_seconds)
        return self.actor


class CollectionPolicy:
    """Reuse the selector with weights and prior synchronized before each episode."""
    def __init__(self, method, collection_device=None):
        self.method = method
        learner_device = getattr(method, 'device', getattr(getattr(method, 'runner', None), 'device', 'cpu'))
        self.mirror = ActorMirror(learner_device, collection_device)
        self.device = self.mirror.device
        self.proxy = method if self.device == torch.device(learner_device) else copy.copy(method)
        if self.proxy is not method:
            if not hasattr(method, 'actor'):
                raise ValueError('separate collection device requires a learned actor')
            self.proxy.device = self.device
            self.proxy.actor = self.mirror.sync(method.actor)

    def begin_episode(self, topology, *, seed):
        actor = getattr(self.method, 'actor', None)
        if actor is not None:
            self.proxy.actor = self.mirror.sync(actor)
        hook = getattr(self.method, 'begin_episode', None)
        if hook is not None:
            hook(topology, seed=seed)

    def select(self, *args, **kwargs):
        return self.proxy.select(*args, **kwargs)
