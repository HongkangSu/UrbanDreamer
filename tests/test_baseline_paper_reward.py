"""Causal mean reward must reach collection, critics and model reward labels."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import train_toy_actor_critic as ac
from tools import train_toy_baseline as trainer
from urban_dreamer.learning.baselines import common
from urban_dreamer.learning.baselines.dreamer import DreamerMethod
from urban_dreamer.learning.baselines.mbpo import MbpoMethod
from urban_dreamer.learning.global_reward import causal_context
from urban_dreamer.learning.replay import ReplayBuffer
from test_toy_baselines import make_args


def paper_args(tmp_path, **options):
    return make_args(tmp_path, reward_objective='observed_mean', global_reward_only=True, **options)


def test_growing_starts_correction_and_unfinished_no_penalty(tmp_path, monkeypatch):
    runner = common.EpisodeRunner(paper_args(tmp_path))
    monkeypatch.setattr(runner.demand.__class__, 'sample', lambda self, **kw: runner.topology.vehicles)
    runner.deadline = 60.
    flow = runner.network.extract_flow({})

    class Session:
        done = False
        travel_times = {}
        microstep_seconds = 1.
        metrics = {'microsteps': 60}
        index = 0

        def __init__(self, *args, **kwargs):
            self.active_vehicle_count = 0
            self.started_vehicle_count = 0
            self.cumulative_vehicle_seconds = 0.

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def state(self, timestamp, starts, cumulative):
            self.active_vehicle_count = starts
            self.started_vehicle_count = starts
            self.cumulative_vehicle_seconds = cumulative
            return dict(time=timestamp, flow=flow, active_vehicle_count=starts,
                started_vehicle_count=starts, cumulative_vehicle_seconds=cumulative,
                vehicles=[dict(vehicle_id=v.vehicle_id, departed=i < starts, arrived=False)
                    for i, v in enumerate(runner.topology.vehicles)])

        def reset(self):
            return self.state(0., 0, 0.)

        def advance_window(self, callback):
            self.index += 1
            state = self.state(30., 1, 60.) if self.index == 1 else self.state(60., 3, 90.)
            return SimpleNamespace(next_state=state, active_vehicle_seconds=60. if self.index == 1 else 30.,
                action_counts=np.zeros(runner.network.edge_count), departure_counts=np.zeros(runner.network.edge_count))

    monkeypatch.setattr(common, 'SumoSession', Session)
    rows, windows, metrics = runner.run_episode('shortest', seed=42, collect=True, record_windows=True)
    assert [w['reward'] for w in windows] == pytest.approx([-1., .5])
    assert metrics['global_reward_sum'] == metrics['selection_score'] == -.5
    assert metrics['observed_mean_seconds'] == 30.
    assert metrics['started_vehicle_count'] == 3
    assert metrics['vehicle_timeout_penalty_sum'] == 0.
    assert metrics['unfinished'] == runner.n
    assert all(r.get('timeout_return', 0.) == 0. for r in rows)
    assert windows[0]['obs'].shape == (runner.network.edge_count * 4 + 4,)
    np.testing.assert_allclose(windows[1]['obs'][-4:], causal_context(30., 60., 1, 1, 60.))


def test_global_observation_independent_of_unrevealed_final_population(tmp_path):
    runner = common.EpisodeRunner(paper_args(tmp_path))
    state = dict(flow=runner.network.extract_flow({}), time=30., active_vehicle_count=2,
        started_vehicle_count=3, cumulative_vehicle_seconds=120., vehicles=())
    first = runner.global_observation(state)
    runner.n += 10000
    np.testing.assert_array_equal(first, runner.global_observation(state))
    np.testing.assert_array_equal(first[-4:], common.accounting_context(state, runner.deadline))


def test_default_critic_context_has_accounting_and_legacy_remains_explicit():
    assert common.reward_objective(SimpleNamespace()) == 'observed_mean'
    assert common.GlobalTrafficCritic(2).network[0].in_features == 12
    assert ac.GlobalTrafficCritic(2).network[0].in_features == 12
    assert common.GlobalTrafficCritic(2, 2).network[0].in_features == 10
    with pytest.raises(ValueError, match='reward_objective'):
        trainer.validate_prior_resume({'global_reward_only': True},
            SimpleNamespace(reward_objective='observed_mean', global_reward_only=True))


def test_model_based_methods_keep_positive_paper_labels(tmp_path):
    runner = common.EpisodeRunner(paper_args(tmp_path))
    flow = runner.network.extract_flow({})
    window = dict(flow=flow, next_flow=flow.copy(), action=np.zeros(runner.network.edge_count),
        departure=np.zeros(runner.network.edge_count), context=causal_context(0., runner.deadline, 1, 1, 60.),
        reward=.5, time=0., terminal=True, active_vehicle_seconds=30., total_vehicle_count=runner.n,
        started_vehicle_count=1, cumulative_vehicle_seconds=60.,
        next_started_vehicle_count=3, next_cumulative_vehicle_seconds=90., vehicles=())
    mbpo = MbpoMethod(runner.args, runner)
    assert mbpo._model_sample(window)['reward'] == .5
    assert mbpo.ensemble[0].trunk[0].in_features == runner.network.edge_count * 10 + 4
    dreamer = DreamerMethod(runner.args, runner)
    dreamer.sequence_buffer = ReplayBuffer(sequence_length=1)
    # No optimizer update is needed to check ingestion of exact model labels.
    dreamer._train_world_model = lambda: {}
    dreamer._train_imagination = lambda: {}
    dreamer.on_episode([], [window], episode=1)
    transition = dreamer.sequence_buffer.episodes[0][0]
    assert transition.global_reward == .5
    assert transition.started_count == 1 and transition.next_started_count == 3
    batch = dreamer.sequence_buffer.sample_batch(1, np.random.default_rng(0))
    assert batch.global_reward.item() == .5
    assert dreamer.latent_critic.net[0].in_features == 292
    assert dreamer.world_model.reward_head[0].in_features == 292


def test_mamba_global_context_uses_started_population(tmp_path):
    from urban_dreamer.learning.baselines.mamba import MambaMethod
    runner = common.EpisodeRunner(paper_args(tmp_path))
    method = MambaMethod(runner.args, runner)
    vehicle = dict(vehicle_id='revealed', current_edge=0, destination_edge=1, departed=True, arrived=False)
    window = dict(flow=runner.network.extract_flow({}), time=30., vehicles=(vehicle,),
        context=causal_context(30., runner.deadline, 1, 3, 120.))
    observed = method._global_observation(window, 10, {'revealed'})
    changed_future = method._global_observation(window, 10000, {'revealed'})
    np.testing.assert_array_equal(observed, changed_future)
    assert observed.shape == (runner.network.edge_count * 10 + 4,)
    assert observed[runner.network.edge_count * 8] == pytest.approx(1. / 3.)


def test_dreamer_imagined_context_keeps_seed_accounting_and_positive_reward(tmp_path, monkeypatch):
    import urban_dreamer.learning.baselines.dreamer as dreamer_module
    from urban_dreamer.learning.replay import MacroTransition
    from test_toy_baselines_model_based import active_snapshot
    runner = common.EpisodeRunner(paper_args(tmp_path))
    method = DreamerMethod(runner.args, runner)
    timestamp = runner.deadline - 30.
    flow = runner.network.extract_flow({})
    seed = MacroTransition(flow=flow, next_flow=flow, action=np.zeros(runner.network.edge_count),
        departure=np.zeros(runner.network.edge_count), context=causal_context(timestamp, runner.deadline, 1, 3, 120.),
        active_vehicle_fraction=.5, vehicles=(active_snapshot(runner),), time=timestamp,
        global_reward=0., started_count=3, next_started_count=3,
        cumulative_vehicle_seconds=120., next_cumulative_vehicle_seconds=120.)
    method.seed_buffer.append_episode([seed], total_vehicle_count=1000, total_travel_minutes=2. / 3.)
    monkeypatch.setattr(method, '_decode_flow', lambda h, z: flow.copy())
    monkeypatch.setattr(method.world_model, 'predict_reward', lambda h, z, context: torch.tensor([.25]))
    monkeypatch.setattr(method, '_prior_dynamics_step', lambda h, z, *args: (h, z))
    monkeypatch.setattr(dreamer_module, 'advance_imagined_window', lambda *args, **kwargs:
        SimpleNamespace(action=np.zeros(runner.network.edge_count), departure=np.zeros(runner.network.edge_count),
            active_vehicle_seconds=30.))
    rows, trajectory = method._imagine_rollout(0)
    latents, next_latents, rewards, terminals = trajectory
    assert rewards == [.25] and terminals == [True]
    np.testing.assert_allclose(latents[0][0, -4:].numpy(), causal_context(timestamp, runner.deadline, 1, 3, 120.))
    np.testing.assert_allclose(next_latents[0][0, -4:].numpy(), causal_context(runner.deadline, runner.deadline, 1, 3, 150.))
    assert not rows


def test_paper_and_explicit_legacy_configs_have_distinct_checkpoint_identity(tmp_path):
    from urban_dreamer.workflow.config import legacy_semantic_fingerprint, load_config
    current = common.EpisodeRunner(paper_args(tmp_path))
    legacy = common.EpisodeRunner(make_args(tmp_path))
    configured = load_config(current.args.config)
    assert current.config.semantic_fingerprint == configured.semantic_fingerprint
    assert legacy.config.semantic_fingerprint == legacy_semantic_fingerprint(configured)
    assert current.config.semantic_fingerprint != legacy.config.semantic_fingerprint


def test_direct_namespaces_cannot_mix_paper_reward_and_timeout_shaping(tmp_path):
    invalid = make_args(tmp_path, reward_objective='observed_mean', global_reward_only=False)
    with pytest.raises(ValueError, match='observed_mean requires global_reward_only=True'):
        common.EpisodeRunner(invalid)
    with pytest.raises(ValueError, match='observed_mean requires global_reward_only=True'):
        ac.RealActorCritic(invalid)
