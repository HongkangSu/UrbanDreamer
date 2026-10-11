"""State and causal update checks without a running SUMO process."""
import copy
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from tools.dynamic_routing_prior import DynamicPriorActor
from tools.train_dynamic_prior_ac import DynamicPriorAC, isolated_rng, parse_args, score, beats_reference, EVALUATION_DEFINITION, set_episode_prior


def fixture_runner():
    torch.set_num_threads(1)
    runner = DynamicPriorAC.__new__(DynamicPriorAC)
    runner.device = torch.device('cpu')
    runner.actor = DynamicPriorActor()
    runner.actor_optimizer = torch.optim.Adam(runner.actor.parameters(), lr=2e-4)
    runner.global_critic = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    runner.global_optimizer = torch.optim.Adam(runner.global_critic.parameters(), lr=3e-4)
    runner.entropy_coefficient = .002
    runner.protocol = dict(episodes=500, profile='od_stagger', prior_weight=1., prior_temperature=30.,
                           prior_schedule='linear', prior_off_episode=250,
                           evaluation_policy='deterministic', evaluation_definition=EVALUATION_DEFINITION)
    return runner


def make_rows(runner):
    features = torch.randn(4, 3, 10)
    mask = torch.ones(4, 3, dtype=torch.bool)
    costs = torch.tensor([[0., 10., 60.]]).repeat(4, 1)
    with torch.no_grad():
        dist = Categorical(logits=runner.actor.forward_logits(features, mask, costs))
        actions = torch.tensor([0, 1, 2, 0])
        old = dist.log_prob(actions)
    rows = [dict(features=features[i].numpy(), mask=mask[i].numpy(), action=int(actions[i]),
                 prior_weight=runner.actor.dynamic_prior_weight,
                 prior_excess_seconds=costs[i].numpy().copy(), old_log_prob=float(old[i]), window=i%2)
            for i in range(4)]
    windows = [dict(obs=np.ones(4)*i, next_obs=np.ones(4)*(i+1), reward=-float(i+1), terminal=i==1)
               for i in range(2)]
    return rows, windows


def test_update_uses_saved_prior_and_trains_head():
    runner = fixture_runner()
    rows, windows = make_rows(runner)
    before = runner.actor.policy_head.weight.detach().clone()
    # There is deliberately no session or live router in this fixture.
    result = runner.update(rows, windows)
    assert result['behavior_logprob_max_error'] < 1e-6
    assert np.isfinite(result['actor_loss'])
    assert result['actor_gradient_norm'] > 0
    assert not torch.equal(before, runner.actor.policy_head.weight)


def test_changed_prior_rows_rejected_before_update():
    runner = fixture_runner()
    rows, windows = make_rows(runner)
    before = copy.deepcopy(runner.actor.state_dict())
    rows[0]['prior_excess_seconds'][0] = 900.
    with pytest.raises(ValueError, match='behavior'):
        runner.update(rows, windows)
    assert all(torch.equal(before[k], v) for k,v in runner.actor.state_dict().items())


def test_checkpoint_roundtrip_rng_and_optimizers(tmp_path):
    runner = fixture_runner()
    runner.update(*make_rows(runner))
    runner.save_checkpoint(tmp_path/'latest.pt', 2, {'best_episode': 0})
    expected = (random.random(), np.random.random(), torch.rand(3))
    restored = fixture_runner()
    episode, progress = restored.load_checkpoint(tmp_path/'latest.pt')
    assert episode == 2 and progress['best_episode'] == 0
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    assert all(torch.equal(v, restored.actor.state_dict()[k]) for k,v in runner.actor.state_dict().items())
    assert restored.actor_optimizer.state_dict()['state']
    assert restored.global_optimizer.state_dict()['state']


@pytest.mark.parametrize('key,value', [('episodes',501), ('profile','original'), ('prior_weight',0.),
                                     ('prior_temperature',60.), ('prior_schedule','constant'),
                                     ('prior_off_episode',251)])
def test_semantic_change_rejects_resume(tmp_path, key, value):
    runner = fixture_runner()
    runner.save_checkpoint(tmp_path/'latest.pt', 1, {})
    runner.protocol[key] = value
    with pytest.raises(ValueError, match='semantic'):
        runner.load_checkpoint(tmp_path/'latest.pt')


def test_eval_rng_isolation():
    torch.manual_seed(10)
    expected = torch.rand(4)
    torch.manual_seed(10)
    with isolated_rng():
        torch.rand(99)
    assert torch.equal(torch.rand(4), expected)


def test_stop_slice_does_not_change_budget():
    args = parse_args(['--config','example.toml','--output-dir','out','--baseline-reference','ref.json','--stop-after','2'])
    assert args.episodes == 500 and args.stop_after == 2


def test_default_evaluation_is_argmax_and_sampling_is_explicit():
    from tools.train_dynamic_prior_ac import evaluation_policy
    command = ['--config', 'example.toml', '--output-dir', 'out', '--baseline-reference', 'ref.json']
    args = parse_args(command)
    assert evaluation_policy(args) == 'deterministic'
    assert args.evaluation_argmax is True
    assert evaluation_policy(parse_args(command + ['--evaluation-sampled'])) == 'sampled_temperature_0.2'
    with pytest.raises(SystemExit):
        parse_args(command + ['--evaluation-temperature', '0'])


def test_sampled_evaluation_repeats_fixed_actions_and_preserves_rng(tmp_path):
    from tools.train_dynamic_prior_ac import evaluate
    draws = []
    def run_episode(**kwargs):
        draws.append((kwargs['policy'], float(torch.rand(())), float(np.random.random()), random.random()))
        return [], dict(vehicle_count=1, unfinished=0, started_vehicle_count=1,
                        cumulative_vehicle_seconds=1., never_departed=0)
    runner = SimpleNamespace(protocol={'evaluation_policy': 'sampled_temperature_2'}, run_episode=run_episode)
    torch.manual_seed(12)
    expected = torch.rand(3)
    torch.manual_seed(12)
    evaluate(runner, tmp_path / 'one', [10, 11])
    evaluate(runner, tmp_path / 'two', [10, 11])
    assert draws[:2] == draws[2:]
    assert all(row[0] == 'sampled_temperature_2' for row in draws)
    assert torch.equal(torch.rand(3), expected)


def test_sampled_actor_only_actions_use_temperature_two_and_skip_prior(monkeypatch):
    from tools.train_dynamic_prior_ac import EpisodeRunner
    runner = fixture_runner()
    runner.args = SimpleNamespace(profile='od_stagger')
    runner.demand = SimpleNamespace(duration_seconds=1800.)
    runner.session_factory = object()
    observer = SimpleNamespace(sample_times=[], timestamp=-1.)
    runner.session_handle = {'session': SimpleNamespace(routing_cost_observer=observer)}
    def forbidden(*args):
        pytest.fail('Sampled actor-only evaluation read a routing prior')
    runner.router = SimpleNamespace(reset=lambda: None, update_costs=forbidden, _distance=forbidden)
    runner.actor.candidate_scorer = torch.nn.Identity()
    runner.actor.policy_head = torch.nn.Linear(10, 1, bias=False)
    with torch.no_grad():
        runner.actor.policy_head.weight.zero_()
        runner.actor.policy_head.weight[0, 0] = 1.
    monkeypatch.setattr(runner.actor, 'forward_logits', forbidden)
    features = np.zeros((1, 3, 10), dtype=np.float32)
    features[0, :, 0] = [1., 3., 100.]
    obs = SimpleNamespace(features=features, mask=np.array([[True, True, False]]))
    monkeypatch.setattr(Categorical, 'sample', lambda self: torch.tensor([0]))
    def collect_episode(self, select, **kwargs):
        actions, aux = select([{'destination_edge': 1}], obs, None, 0.)
        assert actions == [0]  # Explicit sampled action differs from argmax.
        expected = Categorical(logits=torch.tensor([[.5, 1.5, -torch.inf]]))
        assert aux[0]['old_log_prob'] == pytest.approx(expected.log_prob(torch.tensor([0])).item())
        assert not runner.session_handle['observe_costs']
        return [], [], {}
    monkeypatch.setattr(EpisodeRunner, 'run_episode', collect_episode)
    _, metrics = runner.run_episode(policy='sampled_temperature_2', seed=10)
    assert metrics['evaluation_temperature'] == 2.
    assert not metrics['routing_prior_applied']


def test_completion_precedes_att_in_best_selection():
    def row(unfinished, cost):
        return dict(vehicle_count=100, unfinished=unfinished, started_vehicle_count=100,
                    cumulative_vehicle_seconds=cost, never_departed=0)
    reliable, faster = [row(0,200)], [row(1,100)]
    assert score(reliable) > score(faster)
    assert not beats_reference(faster, reliable)


@pytest.mark.parametrize('collect', [False, True])
def test_collection_supplies_complete_archive_metadata(monkeypatch, tmp_path, collect):
    from tools.train_dynamic_prior_ac import EpisodeRunner
    from urban_dreamer.learning.episode_archive import EpisodeArchive
    runner = fixture_runner()
    runner.args = SimpleNamespace(profile='od_stagger')
    runner.demand = SimpleNamespace(duration_seconds=1800.)
    runner.router = SimpleNamespace(reset=lambda: None)
    runner.session_factory = object()
    runner.session_handle = {'session': SimpleNamespace(
        routing_cost_observer=SimpleNamespace(sample_times=[0.]))}
    captured = {}

    def collect_episode(self, select, **kwargs):
        metadata = dict(city='test', method='ac', seed=kwargs['seed'], input_hashes={},
                        total_vehicle_count=0, session_protocol='event', demand={'law': 'old'})
        metadata.update(kwargs['archive_metadata'])
        metadata = kwargs['metadata_transform'](metadata)
        # Exercise the real required-field validation, not a mirrored field list.
        archive = EpisodeArchive(metadata)
        captured.update(archive.metadata)
        return [], [], {}

    monkeypatch.setattr(EpisodeRunner, 'run_episode', collect_episode)
    runner.run_episode(policy='stochastic' if collect else 'deterministic', seed=10042,
                       collect=collect, archive_path=tmp_path/'episode.npz')
    assert captured['episode'] == 10042
    assert captured['phase'] == ('train' if collect else 'evaluation')
    assert captured['demand']['law'] == 'independent_nonhomogeneous_poisson_per_od'


def test_initial_argmax_preserves_subnormalization_cost_difference(monkeypatch):
    from tools.train_dynamic_prior_ac import EpisodeRunner
    runner = fixture_runner()
    runner.args = SimpleNamespace(profile='od_stagger')
    runner.demand = SimpleNamespace(duration_seconds=1800.)
    runner.router = SimpleNamespace(reset=lambda: None, timestamp=0.,
                                   _distance=lambda _: np.array([10. + 1e-8, 10.]))
    runner.session_factory = object()
    runner.session_handle = {'session': SimpleNamespace(routing_cost_observer=SimpleNamespace(
        sample_times=[0.], timestamp=0.))}
    obs = SimpleNamespace(features=np.zeros((1,2,10), dtype=np.float32),
                          mask=np.ones((1,2), dtype=bool), edge_indices=np.array([[0,1]]))

    def collect_episode(self, select, **kwargs):
        actions, aux = select([{'destination_edge': 1}], obs, None, 0.)
        assert actions == [1]
        assert aux[0]['prior_excess_seconds'][0] > 0.
        return [], [], {}

    # Categorical normalization would erase this difference in float32.
    normalized = Categorical(logits=torch.tensor([[-1e-8/30., 0.]])).logits
    assert normalized.argmax(-1).item() == 0
    monkeypatch.setattr(EpisodeRunner, 'run_episode', collect_episode)
    _, metrics = runner.run_episode(policy='deterministic_with_prior', seed=10042, check_initial=True)
    assert metrics['prior_agreement'] == 1.


@pytest.mark.parametrize('costs', [[0., 300., 0.], [300., 0., 1000.]])
def test_actor_only_deterministic_ignores_costs_and_masks_illegal_candidates(monkeypatch, costs):
    from tools.train_dynamic_prior_ac import EpisodeRunner
    runner = fixture_runner()
    runner.args = SimpleNamespace(profile='od_stagger')
    runner.demand = SimpleNamespace(duration_seconds=1800.)
    runner.session_factory = object()
    observer = SimpleNamespace(sample_times=[0.], timestamp=0., values=np.array(costs))
    runner.session_handle = {'session': SimpleNamespace(routing_cost_observer=observer)}

    def forbidden(*args):
        pytest.fail('Actor-only evaluation accessed the dynamic routing prior')

    runner.router = SimpleNamespace(reset=lambda: None, timestamp=-1.,
                                   update_costs=forbidden, _distance=forbidden)
    # Make the actor's neural score explicit, including a larger illegal score.
    runner.actor.candidate_scorer = torch.nn.Identity()
    runner.actor.policy_head = torch.nn.Linear(10, 1, bias=False)
    with torch.no_grad():
        runner.actor.policy_head.weight.zero_()
        runner.actor.policy_head.weight[0, 0] = 1.
    monkeypatch.setattr(runner.actor, 'forward_logits', forbidden)
    features = np.zeros((1, 3, 10), dtype=np.float32)
    features[0, :, 0] = [1., 2., 100.]
    # Static route costs must not add an explicit bias to the neural output.
    features[0, :, 4] = [0., 10., 0.]
    obs = SimpleNamespace(features=features, mask=np.array([[True, True, False]]),
                          edge_indices=np.array([[0, 1, 2]]))

    def collect_episode(self, select, **kwargs):
        actions, aux = select([{'destination_edge': 1}], obs, None, 0.)
        assert actions == [1]
        expected = Categorical(logits=torch.tensor([[1., 2., -torch.inf]]))
        assert aux[0]['old_log_prob'] == pytest.approx(expected.log_prob(torch.tensor([1])).item())
        return [], [], {}

    monkeypatch.setattr(EpisodeRunner, 'run_episode', collect_episode)
    _, metrics = runner.run_episode(policy='deterministic', seed=10042)
    assert metrics['evaluation_policy'] == 'deterministic'
    assert metrics['prior_agreement'] is None


def test_checkpoint_without_actor_only_evaluation_contract_rejects_resume(tmp_path):
    runner = fixture_runner()
    current_protocol = runner.protocol.copy()
    runner.protocol.pop('evaluation_policy')
    runner.protocol.pop('evaluation_definition')
    runner.save_checkpoint(tmp_path/'legacy.pt', 1, {})
    runner.protocol = current_protocol
    with pytest.raises(ValueError, match='semantic'):
        runner.load_checkpoint(tmp_path/'legacy.pt')


@pytest.mark.parametrize('complete_run', [False, True])
@pytest.mark.parametrize('prior_weight', [0., 1.])
def test_initial_best_uses_actor_only_evaluation(monkeypatch, tmp_path, complete_run, prior_weight):
    import tools.train_dynamic_prior_ac as training

    def metric(cost):
        return dict(vehicle_count=10, unfinished=0, started_vehicle_count=10,
                    cumulative_vehicle_seconds=cost, never_departed=0)

    prior, actor_only = [metric(10)], [metric(100)]
    calls, saved, checked = [], [], []
    runner = SimpleNamespace(protocol={}, load_checkpoint=lambda path: None,
                             save_checkpoint=lambda path, episode, progress:
                             saved.append((path.name, copy.deepcopy(progress))))
    runner.run_episode = lambda **kwargs: ([], dict(observed_mean_seconds=10., completion_rate=1.))
    runner.update = lambda rows: dict(actor_loss=0., global_critic_loss=0., learning_seconds=0.)

    def evaluate(self, output, seeds, *, policy='deterministic', check_initial=False):
        calls.append((str(output.relative_to(tmp_path)), policy, check_initial))
        return prior if policy in {'deterministic_with_prior', 'prior'} else actor_only

    monkeypatch.setattr(training, 'DynamicPriorAC', lambda args: runner)
    monkeypatch.setattr(training, 'evaluate', evaluate)
    monkeypatch.setattr(training, 'check_reference', lambda self, rows, path: checked.append(rows))
    monkeypatch.setattr(training, 'set_episode_entropy', lambda self, episode: None)
    monkeypatch.setattr(training, 'set_episode_prior', lambda self, episode: None)
    training.main(['--config', 'unused.toml', '--output-dir', str(tmp_path),
                   '--baseline-reference', 'unused.json', '--episodes', '1' if complete_run else '2',
                   '--stop-after', '1', '--prior-weight', str(prior_weight)])
    reference_policy = 'deterministic_with_prior' if prior_weight else 'prior'
    expected_calls = [('evaluation/initial', reference_policy, bool(prior_weight)),
                      ('evaluation/initial_actor_only', 'deterministic', False)]
    if complete_run:
        expected_calls += [('evaluation/actor_only/episode_000001', 'deterministic', False),
                           ('holdout/prior', 'prior', False),
                           ('holdout/final', 'deterministic', False),
                           ('holdout/best', 'deterministic', False)]
    assert calls == expected_calls
    assert checked == [prior]
    for _, progress in saved:
        assert progress['initial'] == prior
        assert progress['initial_actor_only'] == actor_only
        assert progress['best_score'] == list(score(actor_only))
        assert progress['evaluation_policy'] == 'deterministic'
        assert progress['reference_evaluation_policy'] == reference_policy


def test_actor_only_logits_equal_legacy_dynamic_forward_with_zero_weight():
    from urban_dreamer.models.policy import NextHopActor
    runner = fixture_runner()
    with torch.no_grad():
        runner.actor.policy_head.weight.normal_()
    features = torch.randn(7, 4, 10)
    mask = torch.tensor([[True, False, True, True]]).repeat(7, 1)
    costs = torch.rand(7, 4) * 10000.
    runner.actor.dynamic_prior_weight = 0.
    assert torch.equal(NextHopActor.forward_logits(runner.actor, features, mask),
                       runner.actor.forward_logits(features, mask, costs))


def test_stochastic_collection_keeps_prior_logits_and_frozen_behavior(monkeypatch):
    from tools.train_dynamic_prior_ac import EpisodeRunner
    runner = fixture_runner()
    runner.args = SimpleNamespace(profile='od_stagger')
    runner.demand = SimpleNamespace(duration_seconds=1800.)
    runner.router = SimpleNamespace(reset=lambda: None, timestamp=0.,
                                   _distance=lambda _: np.array([120., 30.]))
    runner.session_factory = object()
    runner.session_handle = {'session': SimpleNamespace(routing_cost_observer=SimpleNamespace(
        sample_times=[0.], timestamp=0.))}
    obs = SimpleNamespace(features=np.zeros((1, 2, 10), dtype=np.float32),
                          mask=np.ones((1, 2), dtype=bool), edge_indices=np.array([[0, 1]]))
    features, mask = torch.from_numpy(obs.features), torch.from_numpy(obs.mask)
    prior = torch.tensor([[90., 0.]])
    expected_distribution = Categorical(logits=runner.actor.forward_logits(features, mask, prior))
    torch.manual_seed(20)
    expected_action = expected_distribution.sample()
    torch.manual_seed(20)

    def collect_episode(self, select, **kwargs):
        actions, aux = select([{'destination_edge': 1}], obs, None, 0.)
        assert actions == expected_action.tolist()
        np.testing.assert_array_equal(aux[0]['prior_excess_seconds'], prior[0].numpy())
        assert aux[0]['old_log_prob'] == expected_distribution.log_prob(expected_action).item()
        return [], [], {}

    monkeypatch.setattr(EpisodeRunner, 'run_episode', collect_episode)
    _, metrics = runner.run_episode(policy='stochastic', seed=10042, collect=True)
    assert metrics['action_policy'] == 'stochastic'
    assert metrics['evaluation_policy'] is None
    assert metrics['routing_prior_applied'] is True


@pytest.mark.parametrize('policy', ['stochastic', 'deterministic_with_prior', 'deterministic', 'prior'])
def test_zero_weight_policy_skips_unused_router_and_preserves_behavior(monkeypatch, policy):
    from tools.train_dynamic_prior_ac import EpisodeRunner
    from urban_dreamer.models.policy import NextHopActor
    runner = fixture_runner()
    runner.actor.dynamic_prior_weight = 0.
    runner.args = SimpleNamespace(profile='od_stagger')
    runner.demand = SimpleNamespace(duration_seconds=1800.)
    runner.session_factory = object()
    runner.session_handle = {'session': SimpleNamespace(routing_cost_observer=SimpleNamespace(
        sample_times=[0.] if policy == 'prior' else [], timestamp=0. if policy == 'prior' else None))}

    def distance(_):
        assert policy == 'prior', 'No-prior policy queried shortest routes'
        return np.array([120., 30., 5.])

    runner.router = SimpleNamespace(reset=lambda: None, timestamp=0., _distance=distance)
    features = np.random.default_rng(42).normal(size=(4, 3, 10)).astype(np.float32)
    mask = np.tile([True, True, False], (4, 1))
    obs = SimpleNamespace(features=features, mask=mask, edge_indices=np.tile([0, 1, 2], (4, 1)))
    expected = Categorical(logits=NextHopActor.forward_logits(
        runner.actor, torch.from_numpy(features), torch.from_numpy(mask)))
    torch.manual_seed(20)
    expected_actions = expected.sample() if policy == 'stochastic' else (
        torch.ones(4, dtype=torch.long) if policy == 'prior' else expected.logits.argmax(-1))
    torch.manual_seed(20)
    captured = {}

    def collect_episode(self, select, **kwargs):
        assert self.session_handle['observe_costs'] is (policy == 'prior')
        captured.update(kwargs['metadata_transform']({'demand': {}}))
        actions, aux = select([{'destination_edge': 1}] * 4, obs, None, 0.)
        assert actions == expected_actions.tolist()
        rows = []
        for i, extra in enumerate(aux):
            np.testing.assert_array_equal(extra['prior_excess_seconds'],
                                          [90., 0., 0.] if policy == 'prior' else [0., 0., 0.])
            assert extra['old_log_prob'] == expected.log_prob(expected_actions)[i].item()
            rows.append(dict(features=features[i], mask=mask[i], action=actions[i], window=i % 2, **extra))
        _, windows = make_rows(runner)
        return rows, windows, {}

    monkeypatch.setattr(EpisodeRunner, 'run_episode', collect_episode)
    rows, metrics = runner.run_episode(policy=policy, seed=10042, collect=policy == 'stochastic')
    assert captured['routing_prior_applied'] is (policy == 'prior')
    assert metrics['routing_prior_applied'] is (policy == 'prior')
    assert metrics['prior_agreement'] == (1. if policy == 'prior' else None)
    if policy == 'stochastic':
        before = runner.actor.policy_head.weight.detach().clone()
        result = runner.update(rows)
        assert result['behavior_logprob_max_error'] == 0.
        assert result['actor_gradient_norm'] > 0.
        assert not torch.equal(before, runner.actor.policy_head.weight)


@pytest.mark.parametrize('prior_weight,policy,observes', [
    (0., 'stochastic', False), (0., 'deterministic', False),
    (0., 'deterministic_with_prior', False), (0., 'prior', True),
    (1., 'stochastic', True), (1., 'deterministic', False),
])
def test_session_reads_costs_only_when_routing_prior_applied(
        monkeypatch, tmp_path, prior_weight, policy, observes):
    import json
    import tools.train_dynamic_prior_ac as training
    network = SimpleNamespace(free_flow_travel_seconds=[10.], edge_ids=['road'])
    args = parse_args(['--config', 'unused.toml', '--output-dir', str(tmp_path),
                       '--baseline-reference', str(tmp_path / 'reference.json'),
                       '--prior-weight', str(prior_weight)])
    (tmp_path / 'reference.json').write_text(json.dumps({'protocol': {
        'input_hashes': {}, 'cost_observation_seconds': 30,
        'demand_profile': {'profile': 'od_stagger'}}}))

    def initialize(self, args):
        self.device = torch.device('cpu')
        self.args = args
        self.network = network
        self.demand = SimpleNamespace(duration_seconds=1800.)
        self.prepared = SimpleNamespace(input_hashes={})
        self.config = SimpleNamespace(semantic_fingerprint='test')
        self.simulation_settings = {}

    monkeypatch.setattr(training.RealActorCritic, '__init__', initialize)
    monkeypatch.setattr(training, 'ReTimedDemand', lambda demand, profile: demand)
    monkeypatch.setattr(training, 'RealTimeShortestRouter', lambda network: SimpleNamespace(reset=lambda: None))
    reads = []

    def read_cost(edge):
        assert observes, 'No-prior episode read dynamic road costs'
        reads.append(edge)
        return 10.

    def initialize_session(self, *args, **kwargs):
        self.network = network
        self._connection = SimpleNamespace(edge=SimpleNamespace(getTraveltime=read_cost))
        self._time_ms = 0

    monkeypatch.setattr(training.SumoEventSession, '__init__', initialize_session)
    monkeypatch.setattr(training.SumoEventSession, 'reset', lambda self: None)
    monkeypatch.setattr(training.SumoEventSession, '_apply_decisions', lambda self, callback, frozen: None)

    def collect_episode(self, select, **kwargs):
        session = kwargs['session_factory']()
        session.reset()
        for second in range(61):
            session._time_ms = second * 1000
            session._apply_decisions(None, None)
        return [], [], {}

    monkeypatch.setattr(training.EpisodeRunner, 'run_episode', collect_episode)
    runner = training.DynamicPriorAC(args)
    _, metrics = runner.run_episode(policy=policy, seed=10042, collect=policy == 'stochastic')
    assert reads == (['road'] * 3 if observes else [])
    assert metrics['cost_observation_times'] == ([0., 30., 60.] if observes else [])
    assert metrics['prior_agreement'] is None
    assert runner.protocol['reference_evaluation_policy'] == (
        'deterministic_with_prior' if prior_weight else 'prior')


@pytest.mark.parametrize('initial_weight', [0., 1., 2.])
def test_linear_prior_schedule_endpoints_and_monotonicity(initial_weight):
    runner = fixture_runner()
    runner.protocol['prior_weight'] = initial_weight
    weights = [set_episode_prior(runner, episode) for episode in range(501)]
    assert weights[0] == weights[1] == initial_weight
    assert weights[249] == pytest.approx(initial_weight / 249)
    assert weights[250:] == [0.] * 251
    assert all(a >= b for a, b in zip(weights, weights[1:]))


def test_constant_prior_schedule_preserves_initial_weight():
    runner = fixture_runner()
    runner.protocol['prior_schedule'] = 'constant'
    assert [set_episode_prior(runner, episode) for episode in [0, 1, 249, 250, 500]] == [1.] * 5


def test_update_rejects_changed_sampling_weight_before_mutating_parameters():
    runner = fixture_runner()
    set_episode_prior(runner, 100)
    rows, windows = make_rows(runner)
    before = copy.deepcopy(runner.actor.state_dict())
    set_episode_prior(runner, 101)
    with pytest.raises(ValueError, match='Frozen behavior prior weight'):
        runner.update(rows, windows)
    assert all(torch.equal(before[k], v) for k, v in runner.actor.state_dict().items())
    set_episode_prior(runner, 100)
    assert runner.update(rows, windows)['behavior_logprob_max_error'] == 0.


@pytest.mark.parametrize('episode', [0, 1, 100, 250, 500])
def test_checkpoint_restores_scheduled_weight_and_next_sample(tmp_path, episode):
    runner = fixture_runner()
    expected_weight = set_episode_prior(runner, episode)
    runner.save_checkpoint(tmp_path / 'latest.pt', episode, {})
    features = torch.randn(4, 3, 10)
    mask = torch.ones(4, 3, dtype=torch.bool)
    costs = torch.tensor([[0., 30., 60.]]).repeat(4, 1)
    expected_actions = Categorical(logits=runner.actor.forward_logits(features, mask, costs)).sample()
    restored = fixture_runner()
    restored.actor.dynamic_prior_weight = 99.
    restored.load_checkpoint(tmp_path / 'latest.pt')
    assert restored.actor.dynamic_prior_weight == expected_weight
    restored_features = torch.randn(4, 3, 10)
    assert torch.equal(features, restored_features)
    actual = Categorical(logits=restored.actor.forward_logits(restored_features, mask, costs)).sample()
    assert torch.equal(expected_actions, actual)
    assert set_episode_prior(restored, episode + 1) == set_episode_prior(runner, episode + 1)


def test_prior_schedule_cli_defaults_and_off_episode_validation():
    argv = ['--config', 'unused.toml', '--output-dir', 'out', '--baseline-reference', 'unused.json']
    args = parse_args(argv)
    assert args.prior_schedule == 'linear' and args.prior_off_episode == 250
    with pytest.raises(SystemExit):
        parse_args(argv + ['--prior-off-episode', '1'])
