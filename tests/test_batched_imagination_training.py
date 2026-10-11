"""The fast tensor groups retain scalar-event objectives and gradients."""
import pytest
import torch
import copy
import numpy as np

from test_urban_dreamer_event_returns import make_case, seed
from urban_dreamer.models.policy import individual_actor_advantage
from urban_dreamer.models.event_critic import EventActionCritic, EVENT_Q_FEATURE_DIM
from urban_dreamer.models.event_critic import build_event_q_inputs, detached_counterfactual_baseline
from urban_dreamer.learning.action_events import ActionEvent, pack_action_events


@pytest.mark.parametrize('deadline', [25., 65., 300.])
def test_grouped_events_match_scalar_returns_losses_and_parameter_gradients(deadline):
    results = []
    for batched in (False, True):
        torch.manual_seed(812)
        trainer, batch, _ = make_case(deadline=deadline, population=4,
            states=[seed('a'), seed('b', remaining_distance=15.)])
        trainer.actor.forward_logits = lambda features, mask: torch.stack((
            trainer.actor.logit.expand(len(features)),
            trainer.actor.logit.new_full((len(features),), -1.)), dim=-1)
        rows, macro = trainer._imagine_from_seed(batch, 0, rollout_index=0, batched_events=batched)
        join = torch.cat if batched else torch.stack
        tensors = {key: join([row[key] for row in rows]) for key in (
            'log_prob', 'entropy', 'executed_q', 'global_return',
            'global_advantage', 'counterfactual_baseline')}
        timeout = (torch.cat([row['timeout_return'] for row in rows]) if batched
                   else torch.tensor([row['timeout_return'] for row in rows]))
        advantage = individual_actor_advantage(tensors['global_advantage'], timeout)
        actor = -(tensors['log_prob'] * advantage.detach() + .002 * tensors['entropy']).mean()
        q = torch.nn.functional.mse_loss(tensors['executed_q'], tensors['global_return'])
        (actor + q + macro).backward()
        parameters = [trainer.actor.logit, trainer.latent_critic.bias,
                      *trainer.event_critic.parameters()]
        gradients = [parameter.grad.detach().clone() for parameter in parameters]
        assert any(torch.count_nonzero(gradient) for gradient in gradients)
        metadata = ([(vehicle, row['decision_time'], index)
                     for row in rows for vehicle, index in zip(row['vehicle_id'], row['event_index'])]
                    if batched else [(row['vehicle_id'], row['decision_time'], row['event_index']) for row in rows])
        results.append((tensors, timeout, gradients, metadata))
    assert results[0][3] == results[1][3]
    for key in results[0][0]:
        torch.testing.assert_close(results[0][0][key], results[1][0][key])
    torch.testing.assert_close(results[0][1], results[1][1])
    for before, after in zip(results[0][2], results[1][2]):
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)


def test_executed_only_critic_forward_matches_full_candidate_loss_gradients():
    torch.manual_seed(39)
    critic = EventActionCritic()
    features = torch.randn(27, 5, EVENT_Q_FEATURE_DIM)
    actions = torch.randint(0, 5, (27,))
    returns = torch.randn(27)
    full = critic(features, torch.ones(27, 5, dtype=torch.bool))
    loss = torch.nn.functional.mse_loss(full.gather(1, actions[:, None]).squeeze(1), returns)
    loss.backward()
    expected = [parameter.grad.clone() for parameter in critic.parameters()]
    critic.zero_grad(set_to_none=True)
    selected = features[torch.arange(27), actions]
    executed = critic(selected[:, None], torch.ones(27, 1, dtype=torch.bool)).squeeze(1)
    reduced_loss = torch.nn.functional.mse_loss(executed, returns)
    reduced_loss.backward()
    torch.testing.assert_close(reduced_loss, loss)
    for before, parameter in zip(expected, critic.parameters()):
        torch.testing.assert_close(parameter.grad, before, rtol=1e-5, atol=1e-6)


def test_window_q_batching_preserves_sampling_conditions_and_earlier_baselines():
    torch.manual_seed(68)
    trainer, _, _ = make_case()
    trainer.event_critic = EventActionCritic()
    latent = torch.randn(1, 4, 64)
    events = [ActionEvent('a', 0, 0., 0, 1, 2), ActionEvent('a', 1, 10., 1, 2, 2)]
    groups = []
    expected = []
    for index, event in enumerate(events):
        width = index + 1  # Exercise unequal candidate widths within one window.
        candidates = np.array([[event.next_edge, 3][:width]])
        features = torch.randn(1, width, 10)
        mask = torch.ones(1, width, dtype=torch.bool)
        context = np.array([.2 + index, .4, 0., 0.], dtype=np.float32)
        probabilities = torch.softmax(torch.randn(1, width), -1)
        groups.append(dict(vehicle_id=['a'], _q_observations=features, _q_mask=mask,
            _q_candidates=candidates, _q_context=context, _q_probabilities=probabilities,
            _q_selected=torch.zeros(1, dtype=torch.long)))
        condition = build_event_q_inputs(trainer.world_model, z=latent, observations=features,
            candidate_edges=torch.tensor(candidates), mask=mask,
            current_edges=torch.tensor([event.current_edge]), destination_edges=torch.tensor([2]),
            decision_times=torch.tensor([event.decision_time]), event_indices=torch.tensor([index]),
            context=torch.tensor(context), events=pack_action_events([[events[:index + 1]]], window_starts=[[0.]]),
            window_start=0.)
        values = trainer.event_critic(condition, mask)
        expected.append((values[:, 0], detached_counterfactual_baseline(probabilities, values, mask)))
    changed_future = copy.deepcopy(groups)
    trainer._complete_window_event_q(groups, events, latent, 0.)
    for group, (value, baseline) in zip(groups, expected):
        torch.testing.assert_close(group['executed_q'], value, rtol=1e-5, atol=2e-7)
        torch.testing.assert_close(group['counterfactual_baseline'], baseline, rtol=1e-5, atol=2e-7)
    changed_future[1]['_q_context'] *= 100
    changed_future[1]['_q_observations'] *= 100
    changed_events = [events[0], ActionEvent('a', 1, 10., 1, 3, 2)]
    trainer._complete_window_event_q(changed_future, changed_events, latent, 0.)
    torch.testing.assert_close(groups[0]['counterfactual_baseline'],
                               changed_future[0]['counterfactual_baseline'], rtol=0, atol=0)


def test_sampling_rejects_nonfinite_actor_outputs_before_drawing_actions():
    trainer, batch, _ = make_case()
    with torch.no_grad():
        trainer.actor.logit.fill_(torch.nan)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='legal actor logits must be finite'):
        trainer._imagine_from_seed(batch, 0, rollout_index=0, batched_events=True)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


def test_direct_multinomial_has_categorical_probabilities_actions_and_rng():
    torch.manual_seed(77)
    logits = torch.randn(3000, 5)
    logits[::3, -1] = -torch.inf
    initial = torch.get_rng_state().clone()
    distribution = torch.distributions.Categorical(logits=logits)
    expected = distribution.sample()
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial)
    probabilities = logits.softmax(-1)
    actual = torch.multinomial(probabilities, 1).squeeze(-1)
    torch.testing.assert_close(probabilities, distribution.probs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
