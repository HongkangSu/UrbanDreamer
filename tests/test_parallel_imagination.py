"""Process scheduling cannot change frozen imagination or its learning objective."""
from types import SimpleNamespace
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from test_event_imagination_protocol import network, seed
from urban_dreamer.traffic.topology import NextHopCandidates
from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.learning.parallel_imagination import (
    collect_parallel_imagination, close_parallel_imagination, _collect_window,
    _initialize_worker, _numpy_state)
from urban_dreamer.models.policy import NextHopActor, CentralizedLatentValueCritic, individual_actor_advantage
from urban_dreamer.models.event_critic import EventActionCritic
from urban_dreamer.models.world_model import EventActionEncoder


class SmallWorld:
    def __init__(self, topology):
        self.dynamics = SimpleNamespace(event_encoder=EventActionEncoder(
            adjacency=torch.tensor(topology.adjacency), static_features=torch.tensor(topology.static_features)),
            graph_prior=SimpleNamespace(rank=1))
        self.calls = []

    def encode(self, flow):
        return flow[..., :1].expand(-1, -1, 64).clone()

    def decode(self, latent):
        result = latent.new_zeros(*latent.shape[:2], 4)
        result[..., 1] = 1.
        return result

    def latent_disagreement(self, latent, action, *, noise, n_samples, action_events, **kwargs):
        self.calls.append((latent.shape[0], noise.detach().clone(), action_events))
        samples = latent[None] + .01 * noise[:, :, 0, None, :]
        variance = samples.var(0, unbiased=True).mean(-1)
        return samples, variance.mean(-1), variance


class Seeds:
    def sample_active_seeds(self, *, batch_size, rng, initial_only):
        assert initial_only
        offsets = rng.integers(0, 3, size=batch_size)
        states = tuple((seed('a', current_edge=1, committed_next_edge=3, pending_decision=True, remaining_distance=10. + offset),
                        seed('b', current_edge=1, committed_next_edge=3, pending_decision=True, remaining_distance=15. + offset))
                       for offset in offsets)
        return SimpleNamespace(vehicles=states, time=torch.zeros(batch_size),
            total_vehicle_count=torch.full((batch_size,), 4), started_count=torch.full((batch_size,), 4),
            cumulative_vehicle_seconds=torch.zeros(batch_size), flow=torch.zeros(batch_size, 1, 4, 8))


def trainer_case(deadline=65., rollouts=8):
    torch.manual_seed(212)
    trainer = MainlineTrainer.__new__(MainlineTrainer)
    trainer.device = torch.device('cpu')
    trainer.network = network()
    adjacency = np.zeros((4, 4), dtype=np.float32)
    for i, successors in enumerate(trainer.network.adjacency_list):
        adjacency[i, list(successors)] = 1.
    trainer.network = replace(trainer.network, adjacency=adjacency)
    trainer.candidates = NextHopCandidates(trainer.network)
    trainer.world_model = SmallWorld(trainer.network)
    trainer.actor = NextHopActor()
    trainer.latent_critic = CentralizedLatentValueCritic()
    trainer.event_critic = EventActionCritic()
    trainer.online_replay = Seeds()
    trainer.rng = np.random.default_rng(124)
    trainer._episode_end_time = lambda mode='online': deadline
    trainer.config = SimpleNamespace(resources=SimpleNamespace(imagination_rollouts_per_pass=rollouts), method=SimpleNamespace(disagreement_samples=3,
        disagreement_weight=.25, fm_steps=2, fm_solver='euler', picard_iterations=None),
        simulation=SimpleNamespace(vehicle_max_speed_mps=10.))
    return trainer


def collect_result(trainer, workers):
    torch.manual_seed(434)
    groups, losses = collect_parallel_imagination(trainer, workers)
    joined = {key: torch.cat([group[key] for group in groups]) for key in
        ('global_return', 'global_advantage', 'counterfactual_baseline', 'executed_q',
         'log_prob', 'entropy', 'timeout_return')}
    advantage = individual_actor_advantage(joined['global_advantage'], joined['timeout_return'])
    actor = -(joined['log_prob'] * advantage.detach() + .002 * joined['entropy']).mean()
    q = torch.nn.functional.mse_loss(joined['executed_q'], joined['global_return'])
    macro = torch.stack(losses).mean()
    (actor + q + macro).backward()
    gradients = [parameter.grad.clone() for model in (trainer.actor, trainer.event_critic, trainer.latent_critic)
                 for parameter in model.parameters()]
    assert all(parameter.grad is None for parameter in trainer.world_model.dynamics.event_encoder.parameters())
    metadata = [(group['vehicle_id'], group['event_index'], group['decision_time'], group['window']) for group in groups]
    return joined, gradients, metadata, torch.get_rng_state(), torch.stack([actor, q, macro]).detach()


def test_one_and_two_workers_preserve_actions_baselines_losses_gradients_and_rng(monkeypatch):
    monkeypatch.setattr('urban_dreamer.learning.training.IMAGINED_HORIZON', 3)
    trainers = [trainer_case(rollouts=2), trainer_case(rollouts=2)]
    try:
        results = [collect_result(trainer, workers) for trainer, workers in zip(trainers, (1, 2))]
        assert results[0][2] == results[1][2]
        for name, value in results[0][0].items():
            torch.testing.assert_close(value, results[1][0][name], rtol=0, atol=0)
        for before, after in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(results[0][3], results[1][3], rtol=0, atol=0)
        torch.testing.assert_close(results[0][4], results[1][4], rtol=0, atol=0)
        assert len(trainers[0].world_model.calls) == 2  # No model transition for the five-second tail.
        assert all(call[0] == 2 for call in trainers[0].world_model.calls)
        assert all(torch.isfinite(value).all() for value in results[0][0].values())
    finally:
        for trainer in trainers:
            close_parallel_imagination(trainer)


def test_parallel_partial_tail_matches_serial_event_objective(monkeypatch):
    trainer = trainer_case(deadline=5., rollouts=1)
    # Derive the exact explicit child RNG seed from the checkpointed parent RNG.
    torch.manual_seed(434)
    actor_seed = torch.randint(0, 2**63 - 1, (1, 2))[0, 0].item()
    batch = trainer.online_replay.sample_active_seeds(batch_size=1, rng=np.random.default_rng(124), initial_only=True)
    torch.manual_seed(actor_seed)
    expected, expected_macro = trainer._imagine_from_seed(batch, 0, rollout_index=0, batched_events=True)
    generator_state = torch.get_rng_state().clone()
    try:
        torch.manual_seed(434)
        actual, actual_macro = collect_parallel_imagination(trainer, 1)
        assert not trainer.world_model.calls
        assert [row['event_index'] for row in actual] == [row['event_index'] for row in expected]
        for key in ('global_return', 'global_advantage', 'counterfactual_baseline', 'executed_q',
                    'log_prob', 'entropy', 'timeout_return'):
            torch.testing.assert_close(torch.cat([row[key] for row in actual]),
                                       torch.cat([row[key] for row in expected]), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_macro[0], expected_macro, rtol=1e-5, atol=1e-6)
        assert all((group['timeout_return'] == 0).all() for group in actual)
        assert generator_state.numel() > 0
    finally:
        close_parallel_imagination(trainer)


def test_worker_weight_versions_refresh_baseline_and_do_not_mutate_saved_values():
    trainer = trainer_case()
    encoder = trainer.world_model.dynamics.event_encoder
    _initialize_worker(trainer.candidates, trainer.network.adjacency.copy(), trainer.network.static_features.copy())
    batch = trainer.online_replay.sample_active_seeds(batch_size=1, rng=trainer.rng, initial_only=True)
    task = dict(version=1, weights=dict(actor=_numpy_state(trainer.actor), critic=_numpy_state(trainer.event_critic),
        encoder=_numpy_state(encoder), prior=(0., 30.)), states=copy.deepcopy(batch.vehicles[0]),
        rng=torch.Generator().manual_seed(2).get_state().numpy(), decoded=trainer.world_model.decode(torch.zeros(1, 4, 64))[0].numpy(),
        latent=np.zeros((4, 64), dtype=np.float32), population=4, total_vehicle_count=4, cumulative_vehicle_seconds=0., deadline=5., start=0., max_speed=10., event_count=0)
    before = _collect_window(copy.deepcopy(task))
    saved = before['baseline'].copy()
    task['version'] += 1
    task['weights']['critic']['scorer.4.bias'] += 2.
    after = _collect_window(task)
    np.testing.assert_allclose(after['baseline'], saved + 2., rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(before['baseline'], saved)
    np.testing.assert_array_equal(before['selected'], after['selected'])


def test_invalid_worker_count_rejected_before_sampling():
    trainer = trainer_case()
    for count in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            collect_parallel_imagination(trainer, count)


@pytest.mark.parametrize('deadline,horizon', [(65., 3), (300., 2)])
def test_multiwindow_parallel_matches_serial_returns_bootstrap_and_gradients(monkeypatch, deadline, horizon):
    monkeypatch.setattr('urban_dreamer.learning.training.IMAGINED_HORIZON', horizon)
    serial, parallel = trainer_case(deadline, rollouts=2), trainer_case(deadline, rollouts=2)
    batch = serial.online_replay.sample_active_seeds(batch_size=2, rng=serial.rng, initial_only=True)
    torch.manual_seed(434)
    seeds = torch.randint(0, 2**63 - 1, (2, 2)).tolist()
    reference, losses = [], []
    for index, (actor_seed, noise_seed) in enumerate(seeds):
        torch.manual_seed(actor_seed)
        noise_generator = torch.Generator().manual_seed(noise_seed)
        def predict(latent, action, action_events):
            noise = torch.randn((3, 1, 1, 64), generator=noise_generator)
            samples, variance, _ = serial.world_model.latent_disagreement(latent,
                torch.as_tensor(action)[None], noise=noise, n_samples=3, action_events=action_events)
            return samples[0], variance.reshape(())
        serial._predict_with_disagreement = predict
        rows, loss = serial._imagine_from_seed(batch, index, rollout_index=index, batched_events=True)
        reference.extend(rows)
        losses.append(loss)
    expected = {key: torch.cat([row[key] for row in reference]) for key in (
        'global_return', 'global_advantage', 'counterfactual_baseline', 'executed_q',
        'log_prob', 'entropy', 'timeout_return')}
    advantage = individual_actor_advantage(expected['global_advantage'], expected['timeout_return'])
    objective = (torch.stack(losses).mean()
        + torch.nn.functional.mse_loss(expected['executed_q'], expected['global_return'])
        - (expected['log_prob'] * advantage.detach() + .002 * expected['entropy']).mean())
    objective.backward()
    expected_gradients = [parameter.grad.clone() for model in (serial.actor, serial.event_critic, serial.latent_critic)
                          for parameter in model.parameters()]
    try:
        actual = collect_result(parallel, 2)
        for key, value in expected.items():
            torch.testing.assert_close(actual[0][key], value, rtol=1e-5, atol=1e-6)
        for before, after in zip(expected_gradients, actual[1]):
            torch.testing.assert_close(before, after, rtol=1e-4, atol=2e-6)
        torch.testing.assert_close(objective.detach(), actual[-1].sum(), rtol=1e-5, atol=1e-6)
        assert any(torch.count_nonzero(gradient) for gradient in actual[1])
    finally:
        close_parallel_imagination(parallel)


def test_empty_event_windows_keep_macro_loss_and_full_work_counters(monkeypatch):
    monkeypatch.setattr('urban_dreamer.learning.training.IMAGINED_HORIZON', 2)
    trainer = trainer_case(deadline=100., rollouts=1)
    sample = trainer.online_replay.sample_active_seeds
    def no_events(**kwargs):
        batch = sample(**kwargs)
        for states in batch.vehicles:
            for state in states:
                state.update(pending_decision=False, remaining_distance=10000., committed_next_edge=3)
        return batch
    trainer.online_replay.sample_active_seeds = no_events
    try:
        groups, losses = collect_parallel_imagination(trainer, 1)
        assert groups == [] and len(losses) == 1
        losses[0].backward()
        assert any(parameter.grad is not None for parameter in trainer.latent_critic.parameters())
        assert all(parameter.grad is None for parameter in trainer.actor.parameters())
        assert trainer._last_parallel_work == dict(rollouts=1, windows=2, seconds=60., active_vehicle_seconds=120.)
    finally:
        close_parallel_imagination(trainer)


def test_worker_failure_closes_pool_without_returning_partial_training_batch(monkeypatch):
    trainer = trainer_case()
    sample = trainer.online_replay.sample_active_seeds
    def invalid(**kwargs):
        batch = sample(**kwargs)
        batch.vehicles[0][0]['committed_next_edge'] = 999
        return batch
    trainer.online_replay.sample_active_seeds = invalid
    with pytest.raises(ValueError, match='illegal committed next hop'):
        collect_parallel_imagination(trainer, 1)
    assert not hasattr(trainer, '_parallel_imagination_executor')
    for model in (trainer.actor, trainer.event_critic, trainer.latent_critic):
        assert all(parameter.grad is None for parameter in model.parameters())
