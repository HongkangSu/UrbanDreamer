"""Model-based baselines train route actions against observed physical turns."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_route_imagination import topology, seed
from urban_dreamer.learning.baselines.dreamer import DreamerMethod
from urban_dreamer.learning.baselines.mbpo import MbpoMethod
from urban_dreamer.learning.baselines.mamba import MambaMethod
from urban_dreamer.learning.global_reward import causal_context
from urban_dreamer.learning.imagination import advance_imagined_window
from urban_dreamer.learning.observations import build_actor_observations, selected_decisions
from urban_dreamer.models.policy import ActionSelection
from urban_dreamer.traffic.environment import RouteDecision
from urban_dreamer.traffic.route_candidates import RouteCandidates


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def make_method(name, traffic_model='meso'):
    torch.manual_seed(21)
    args = SimpleNamespace(reward_objective='observed_mean', global_reward_only=True,
        seed=42, actor_lr=3e-4, critic_lr=3e-4, step_seconds=1.,
        mamba_model_warmup_updates=0, mamba_min_model_frames=0, mamba_burn_in_windows=0)
    network = topology()
    runner = SimpleNamespace(device=torch.device('cpu'), network=network,
        candidates=RouteCandidates(network, traffic_model=traffic_model), deadline=60., n=1,
        config=SimpleNamespace(simulation=SimpleNamespace(traffic_model=traffic_model,
            vehicle_max_speed_mps=10.)))
    return {'dreamer': DreamerMethod, 'mbpo': MbpoMethod, 'mamba': MambaMethod}[name](args, runner)


def collect_toy_episode(method):
    """Collect actual route samples while the physical engine emits its turn stream."""
    runner = method.runner
    if isinstance(method, MambaMethod):
        method.begin_episode(SimpleNamespace(vehicles=[SimpleNamespace(vehicle_id='v')]), seed=42)
    states, flow = [seed()], runner.network.extract_flow({})
    rows, windows, cumulative = [], [], 0.
    for wi in range(2):
        start = float(wi * 30)
        before = deepcopy(states)
        context = causal_context(timestamp=start, deadline=60., active_count=1,
            started_count=1, cumulative_seconds=cumulative)
        def decide(pending, frozen, now):
            obs = build_actor_observations(policy_edge_features=flow[:, [2, 1]],
                vehicle_states=pending, candidates=runner.candidates,
                episode_progress=now / 60., active_vehicle_fraction=1.)
            choices, aux = method.select(pending, obs, flow, now)
            for index, state in enumerate(pending):
                rows.append(dict(vehicle_id=state['vehicle_id'], window=wi,
                    current_edge=state['current_edge'], features=obs.features[index].copy(),
                    mask=obs.mask[index].copy(), edge_indices=obs.edge_indices[index].copy(),
                    candidate_routes=obs.routes[index], action=choices[index], **aux[index]))
            return selected_decisions(pending, obs, choices, now)
        result = advance_imagined_window(states, candidates=runner.candidates,
            edge_speeds_mps=np.full(5, 10.), frozen_flow=flow, start_time=start,
            total_vehicle_count=1, decide=decide, session_protocol='event',
            traffic_model=runner.config.simulation.traffic_model)
        cumulative += result.active_vehicle_seconds
        next_context = causal_context(timestamp=start + 30., deadline=60., active_count=1,
            started_count=1, cumulative_seconds=cumulative)
        windows.append(dict(flow=flow.copy(), next_flow=flow.copy(),
            action=result.action.copy(), departure=result.departure.copy(),
            context=context, next_context=next_context,
            obs=np.concatenate((flow[:, :4].reshape(-1), context)),
            next_obs=np.concatenate((flow[:, :4].reshape(-1), next_context)),
            vehicles=before, time=start, end_time=start + 30., terminal=wi == 1,
            reward=-result.active_vehicle_seconds / 60., total_vehicle_count=1,
            active_vehicle_seconds=result.active_vehicle_seconds,
            started_vehicle_count=1, next_started_vehicle_count=1,
            started_count=1, next_started_count=1,
            cumulative_vehicle_seconds=cumulative - result.active_vehicle_seconds,
            next_cumulative_vehicle_seconds=cumulative,
            action_events=result.action_events))
    return rows, windows


def capture_motion(monkeypatch, implementation):
    captured = []
    def advance(states, **kwargs):
        result = advance_imagined_window(states, **kwargs)
        captured.append((deepcopy(states), result))
        return result
    monkeypatch.setattr(implementation, 'advance_imagined_window', advance)
    return captured


def fixed_mamba_flow(method, monkeypatch):
    flow = method.runner.network.extract_flow({})
    def decode(features):
        out = features.new_zeros((*features.shape[:-1], method.global_dim))
        out[..., :40] = torch.as_tensor(flow.reshape(-1), device=features.device)
        return out
    monkeypatch.setattr(method.global_model.observation_decoder, 'forward', decode)


@pytest.mark.parametrize('name', ['dreamer', 'mbpo', 'mamba'])
@pytest.mark.parametrize('column', [0, 1])
def test_actual_imagination_preserves_same_first_hop_route_choice_and_turn_inputs(monkeypatch, name, column):
    implementation = __import__(f'urban_dreamer.learning.baselines.{name}', fromlist=['x'])
    method = make_method(name)
    rows, windows = collect_toy_episode(method)
    capture = capture_motion(monkeypatch, implementation)
    flow = windows[0]['flow']
    conditions = []
    if name == 'dreamer':
        monkeypatch.setattr(implementation, 'IMAGINED_HORIZON', 1)
        # Populate the real seed replay without running a separate learner pass.
        from urban_dreamer.learning.replay import MacroTransition
        method.seed_buffer.append_episode([MacroTransition(flow=flow, next_flow=flow,
            action=windows[0]['action'], departure=windows[0]['departure'],
            context=windows[0]['context'], active_vehicle_fraction=1.,
            vehicles=windows[0]['vehicles'], time=0., episode=0, step_index=0,
            started_count=1, cumulative_vehicle_seconds=0., next_started_count=1,
            next_cumulative_vehicle_seconds=30.)],
            total_vehicle_count=1, total_travel_minutes=1.)
        monkeypatch.setattr(method, '_decode_flow', lambda h, z: flow.copy())
        monkeypatch.setattr(method, '_sample_imagined_actions',
            lambda actor, features, mask: [column] * len(features))
        original = method._prior_dynamics_step
        def step(h, z, action, departure, context):
            conditions.append(action.copy())
            return original(h, z, action, departure, context)
        monkeypatch.setattr(method, '_prior_dynamics_step', step)
        imagined, _ = method._imagine_rollout(0)
        assert len(imagined) == 1 and imagined[0]['mask'].shape == (4,)
        assert imagined[0]['candidate_routes'][column] == ((0, 1, 2, 4) if column == 0 else (0, 1, 3, 4))
    elif name == 'mbpo':
        monkeypatch.setattr(implementation, 'ROLLOUT_LENGTH', 1)
        monkeypatch.setattr(method.actor, 'sample_action', lambda features, mask:
            ActionSelection(torch.full((len(features),), column, dtype=torch.long),
                torch.zeros(len(features)), torch.zeros(len(features))))
        original = method._predict_transition
        def predict(flow, action, departure, context):
            conditions.append(action.copy())
            return original(flow, action, departure, context)
        monkeypatch.setattr(method, '_predict_transition', predict)
        imagined = method._rollout_from_seed(windows[0], 0)
        assert len(imagined) == 1 and imagined[0]['mask'].shape == (4,)
        assert imagined[0]['edge_indices'][column] == 1
    else:
        monkeypatch.setattr(implementation, 'IMAGINATION_HORIZON', 1)
        fixed_mamba_flow(method, monkeypatch)
        original = method.actor.routing_distribution
        def distribution(latent, candidates, mask, **kwargs):
            result = original(latent, candidates, mask, **kwargs)
            result.sample = lambda: torch.full((len(mask),), column, dtype=torch.long)
            return result
        monkeypatch.setattr(method.actor, 'routing_distribution', distribution)
        data = method._pack_episode(rows, windows, 0)
        initial = method.global_model.initial_state(1, 1, device=method.device)
        hook = method.global_model.transition.register_forward_pre_hook(
            lambda _, arguments: conditions.append(arguments[0].detach().cpu().numpy().reshape(-1).copy()))
        imagined, _, _ = method._imagined_rollout(data, initial, 0, {'v'})
        hook.remove()
        assert len(imagined) == 1 and imagined[0]['time'] == 0.
        assert imagined[0]['mask'].shape == (1, 4)
        assert imagined[0]['candidate_routes'][0][column] == ((0, 1, 2, 4) if column == 0 else (0, 1, 3, 4))
    assert len(capture) == 1
    states, physical = capture[0]
    assert len(physical.decisions) == 1 and isinstance(physical.decisions[0], RouteDecision)
    assert physical.decisions[0].route == ((0, 1, 2, 4) if column == 0 else (0, 1, 3, 4))
    assert [(event.current_edge, event.next_edge, event.decision_time)
            for event in physical.action_events] == [(1, 2 + column, 1.)]
    assert states[0]['current_edge'] == 4 and states[0]['committed_route'] == (4,)
    if name == 'mamba':
        np.testing.assert_array_equal(conditions[-1][5:10], physical.action)
        np.testing.assert_array_equal(conditions[-1][:5], [0., 1., 0., 0., 0.])
    else:
        np.testing.assert_array_equal(conditions[-1], physical.action)
    assert conditions[-1][1 if name != 'mamba' else 6] == 0.  # Policy first hop is not the model turn input.


@pytest.mark.parametrize('name', ['dreamer', 'mbpo', 'mamba'])
def test_route_policy_and_model_updates_have_finite_gradients(monkeypatch, name):
    implementation = __import__(f'urban_dreamer.learning.baselines.{name}', fromlist=['x'])
    for constant in ('WORLD_MODEL_UPDATES_PER_EPISODE', 'IMAGINATION_PASSES', 'IMAGINATION_ROLLOUTS',
                     'MODEL_TRAIN_ROUNDS', 'POLICY_UPDATES_PER_EPISODE', 'WORLD_UPDATES', 'PPO_EPOCHS'):
        if hasattr(implementation, constant):
            monkeypatch.setattr(implementation, constant, 1)
    method = make_method(name)
    rows, windows = collect_toy_episode(method)
    assert len(rows) == 1 and rows[0]['mask'].shape == (4,)
    if name == 'mbpo':
        windows = windows[:1]  # This seed still needs a route policy decision.
    if name == 'dreamer':
        monkeypatch.setattr(method, '_decode_flow', lambda h, z: windows[0]['flow'].copy())
    elif name == 'mamba':
        fixed_mamba_flow(method, monkeypatch)
    before = [parameter.detach().clone() for parameter in method.actor.parameters()]
    metrics = method.on_episode(rows, windows, episode=1)
    assert metrics['imagined_events'] > 0
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, method.actor.parameters()))
    models = [method.world_model] if name == 'dreamer' else [method.ensemble] if name == 'mbpo' else [method.world_model, method.global_model]
    for model in models + [method.actor]:
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_mamba_real_global_model_conditions_on_observed_cohort_turns(monkeypatch):
    method = make_method('mamba')
    rows, windows = collect_toy_episode(method)
    data = method._pack_episode(rows, windows, 0)
    actions = []
    hook = method.global_model.transition.register_forward_pre_hook(
        lambda _, arguments: actions.append(arguments[0].detach().cpu().numpy().reshape(-1).copy()))
    method._global_sequence(data, 0, train=False, cohort={'v'})
    hook.remove()
    # observe() also calls transition(); every second call is the consequence.
    np.testing.assert_array_equal(actions[1][5:10], windows[0]['action'])
    np.testing.assert_array_equal(actions[1][:5], [0., 1., 0., 0., 0.])
    np.testing.assert_array_equal(actions[3][:10], np.zeros(10))
    assert method.learning_protocol()['local_clock'] == 'route_decision_30_seconds'


@pytest.mark.parametrize('name', ['dreamer', 'mbpo', 'mamba'])
def test_route_imagination_also_uses_event_clock_with_micro_physics(monkeypatch, name):
    method = make_method(name, traffic_model='micro')
    assert getattr(method, 'microstep_seconds', 1.) == 1.
    rows, windows = collect_toy_episode(method)
    implementation = __import__(f'urban_dreamer.learning.baselines.{name}', fromlist=['x'])
    for constant in ('WORLD_MODEL_UPDATES_PER_EPISODE', 'IMAGINATION_PASSES', 'IMAGINATION_ROLLOUTS',
                     'MODEL_TRAIN_ROUNDS', 'POLICY_UPDATES_PER_EPISODE', 'WORLD_UPDATES', 'PPO_EPOCHS'):
        if hasattr(implementation, constant):
            monkeypatch.setattr(implementation, constant, 1)
    capture = capture_motion(monkeypatch, implementation)
    if name == 'dreamer':
        monkeypatch.setattr(method, '_decode_flow', lambda h, z: windows[0]['flow'].copy())
    elif name == 'mamba':
        fixed_mamba_flow(method, monkeypatch)
    method.on_episode(rows, windows, episode=1)
    assert capture and all(len(result.decisions) <= 1 for _, result in capture)
