"""Route choices reach policy learning while physical turns reach the frozen WM."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
import pytest

from test_parallel_imagination import SmallWorld, trainer_case, collect_result
from test_route_imagination import topology
from test_event_imagination_protocol import seed
from urban_dreamer.learning.parallel_imagination import collect_parallel_imagination, close_parallel_imagination
from urban_dreamer.traffic.route_candidates import RouteCandidates


class RouteSeeds:
    def sample_active_seeds(self, *, batch_size, rng, initial_only):
        assert initial_only
        offsets = rng.integers(0, 3, size=batch_size)
        states = tuple((seed('a', destination_edge=4, remaining_distance=10. + offset,
                             committed_next_edge=1, pending_decision=False),)
                       for offset in offsets)
        return SimpleNamespace(vehicles=states, time=torch.zeros(batch_size),
            total_vehicle_count=torch.ones(batch_size, dtype=torch.long),
            started_count=torch.ones(batch_size, dtype=torch.long),
            cumulative_vehicle_seconds=torch.zeros(batch_size),
            flow=torch.zeros(batch_size, 1, 5, 8))


def route_trainer(rollouts=1, deadline=65.):
    trainer = trainer_case(deadline=deadline, rollouts=rollouts)
    network = topology((10., 1000., 1000., 1000., 1000.))
    adjacency = np.zeros((5, 5), dtype=np.float32)
    for edge, successors in enumerate(network.adjacency_list):
        adjacency[edge, list(successors)] = 1.
    trainer.network = replace(network, adjacency=adjacency)
    trainer.candidates = RouteCandidates(trainer.network)
    trainer.world_model = SmallWorld(trainer.network)
    trainer.online_replay = RouteSeeds()
    trainer.config.simulation.routing_mode = 'current_to_dest'
    return trainer


def test_route_serial_and_parallel_train_policy_q_on_choices_and_world_model_on_turns(monkeypatch):
    monkeypatch.setattr('urban_dreamer.learning.training.IMAGINED_HORIZON', 3)
    trainer = route_trainer()
    torch.manual_seed(434)
    actor_seed, noise_seed = torch.randint(0, 2**63 - 1, (1, 2))[0].tolist()
    batch = trainer.online_replay.sample_active_seeds(batch_size=1,
        rng=np.random.default_rng(124), initial_only=True)
    torch.manual_seed(actor_seed)
    noise_generator = torch.Generator().manual_seed(noise_seed)
    def predict(latent, action, action_events):
        noise = torch.randn((3, 1, 1, 64), generator=noise_generator)
        samples, variance, _ = trainer.world_model.latent_disagreement(latent,
            torch.as_tensor(action)[None], noise=noise, n_samples=3, action_events=action_events)
        return samples[0], variance.reshape(())
    trainer._predict_with_disagreement = predict
    expected, expected_macro = trainer._imagine_from_seed(batch, 0,
        rollout_index=0, batched_events=True)
    # The first path choice is on singleton edge 0; WM receives the later
    # observed physical fork 1, with no second policy sample at that time.
    assert [row['decision_time'] for row in expected] == [0., 30., 60.]
    assert trainer.world_model.calls[0][2].current_edge.tolist() == [1]
    try:
        torch.manual_seed(434)
        actual, actual_macro = collect_parallel_imagination(trainer, 1)
        assert [row['decision_time'] for row in actual] == [0., 30., 60.]
        for key in ('global_return', 'global_advantage', 'counterfactual_baseline',
                    'executed_q', 'log_prob', 'entropy', 'timeout_return'):
            torch.testing.assert_close(torch.cat([row[key] for row in actual]),
                torch.cat([row[key] for row in expected]), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_macro[0], expected_macro, rtol=1e-5, atol=1e-6)
    finally:
        close_parallel_imagination(trainer)


def test_route_parallel_worker_count_preserves_losses_gradients_and_rng(monkeypatch):
    monkeypatch.setattr('urban_dreamer.learning.training.IMAGINED_HORIZON', 3)
    trainers = [route_trainer(rollouts=2), route_trainer(rollouts=2)]
    try:
        first, second = [collect_result(trainer, workers)
                         for trainer, workers in zip(trainers, (1, 2))]
        assert first[2] == second[2]
        for key in first[0]:
            torch.testing.assert_close(first[0][key], second[0][key], rtol=0, atol=0)
        for left, right in zip(first[1], second[1]):
            assert torch.isfinite(left).all()
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(first[3], second[3], rtol=0, atol=0)
    finally:
        for trainer in trainers:
            close_parallel_imagination(trainer)


def test_mainline_route_collection_reuses_full_trainer_state_and_archives_compatible_turns(tmp_path):
    pytest.importorskip('libsumo')
    from test_urban_dreamer_training import make_trainer, fixed_demand
    from test_route_checkpoint_reuse import route_config
    from urban_dreamer.learning.training import MainlineTrainer
    from urban_dreamer.learning.episode_archive import read_episode_archive, to_replay_episode
    from tools.run_online_pilot import routing_world_model_proof

    source = make_trainer(tmp_path / 'source', traffic_model='meso')
    config = route_config(source.config)
    config = replace(config, run=replace(config.run, output_dir=tmp_path / 'route'))
    target = MainlineTrainer(config=config, prepared_inputs=source.prepared_inputs,
                            topology=source.topology)
    try:
        state = source.state_dict()
        target.load_state_dict(state)
        checkpoint = SimpleNamespace(training_state=state)
        assert routing_world_model_proof(target, checkpoint)['world_model_tensors_unchanged']
        vehicles = tuple(replace(vehicle, departure_second=0.) for vehicle in target.topology.vehicles)
        target.demand = fixed_demand(target, lambda **kwargs: vehicles)
        episode = target._run_episode(mode='online', episode_index=0)
        data = read_episode_archive(config.run.output_dir / 'episodes/online/episode_000000.npz')
        assert data['metadata']['routing_mode'] == 'current_to_dest'
        assert data['metadata']['physics_identity']['traffic_model'] == 'meso'
        assert data['policy_decisions']
        assert all(row['decision_time'] % 30 == 0 for row in data['policy_decisions'])
        assert all(row['route'][0] == row['current_edge']
                   and row['route'][-1] == row['destination_edge'] for row in data['policy_decisions'])
        replay = to_replay_episode(config.run.output_dir / 'episodes/online/episode_000000.npz',
                                   expected_session_protocol='event')
        assert sum(len(row.action_events) for row in replay['transitions']) == len(data['decisions'])
        assert episode.metrics['online_policy_decisions'] == len(data['policy_decisions'])
        assert routing_world_model_proof(target, checkpoint)['world_model_tensors_unchanged']
    finally:
        source.close()
        target.close()
