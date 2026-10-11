"""Bounded returns, target lag, and auxiliary isolation across rollout paths."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.models.policy import CentralizedLatentValueCritic
from urban_dreamer.models.event_critic import EventActionCritic
from test_urban_dreamer_event_returns import make_case, seed


def bound(trainer):
    trainer.config.method.critic_value_mode = 'bounded_cohort_v1'
    trainer.config.method.critic_target_tau = .25
    trainer.config.method.critic_terminal_aux_ratio = .25
    trainer.config.method.disagreement_weight = 0.
    trainer.latent_critic = CentralizedLatentValueCritic(bounded=True)
    trainer.target_latent_critic = copy.deepcopy(trainer.latent_critic).eval().requires_grad_(False)
    trainer.event_critic = EventActionCritic(bounded=True)
    trainer._critic_aux_rng = np.random.default_rng(41)
    trainer._critic_aux_torch_cpu = torch.Generator().manual_seed(42).get_state()
    trainer._critic_aux_torch_cuda = []
    return trainer


def test_bootstrap_uses_target_and_ema_updates_full_network():
    trainer, _, _ = make_case()
    bound(trainer)
    latent, context = torch.randn(2, 3, 64), torch.zeros(2, 4)
    with torch.no_grad():
        for parameter in trainer.latent_critic.parameters():
            parameter.add_(.01)
    target_before = [p.clone() for p in trainer.target_latent_critic.parameters()]
    old = trainer._value_prediction(latent, context, [1., 2.], target=True)
    torch.testing.assert_close(old, trainer.target_latent_critic(latent, context, budget=torch.tensor([1., 2.])))
    assert not old.requires_grad
    trainer._update_target_critic()
    for before, online, target in zip(target_before, trainer.latent_critic.parameters(), trainer.target_latent_critic.parameters()):
        torch.testing.assert_close(target, .75 * before + .25 * online)
    assert not any(p.requires_grad for p in trainer.target_latent_critic.parameters())


def test_serial_values_and_event_q_receive_decreasing_physical_budgets():
    trainer, batch, _ = make_case(deadline=60., states=[seed('a'), seed('b', remaining_distance=20.)], population=4)
    bound(trainer)
    calls = {'v': [], 'target': [], 'q': []}
    for name, model in [('v', trainer.latent_critic), ('target', trainer.target_latent_critic), ('q', trainer.event_critic)]:
        model.register_forward_pre_hook(lambda m, a, kw, key=name: calls[key].append(kw['budget'].clone()), with_kwargs=True)
    rows, loss = trainer._imagine_from_seed(batch, 0, rollout_index=0, batched_events=True)
    assert calls['v'][0].item() == pytest.approx(.5)
    assert calls['target'][-1].item() == 0.
    assert calls['q']
    assert min(torch.cat(calls['q'])).item() < .5
    for row in rows:
        assert (row['executed_q'] <= 0).all()
        assert (row['global_return'] <= 0).all()
    loss.backward()
    assert any(p.grad is not None for p in trainer.latent_critic.parameters())
    assert all(p.grad is None for p in trainer.target_latent_critic.parameters())


@pytest.mark.parametrize('fail', [False, True])
def test_auxiliary_keeps_primary_rng_and_actor_q_graphs_untouched(fail):
    trainer, batch, _ = make_case(deadline=60.)
    bound(trainer)
    trainer.rng = np.random.default_rng(8)
    trainer.online_replay = SimpleNamespace(sample_terminal_seeds=lambda **kwargs: copy.deepcopy(batch))
    original = trainer._imagine_from_seed
    def imagine(*args, **kwargs):
        torch.rand(3)
        if fail:
            raise RuntimeError('injected')
        return original(*args, **kwargs)
    trainer._imagine_from_seed = imagine
    cpu_before = torch.get_rng_state().clone()
    numpy_before = copy.deepcopy(trainer.rng.bit_generator.state)
    aux_before = trainer._critic_aux_torch_cpu.clone()
    if fail:
        with pytest.raises(RuntimeError, match='injected'):
            trainer._terminal_aux_losses(4)
    else:
        losses = trainer._terminal_aux_losses(4)
        assert len(losses) == 1
        losses[0].backward()
        assert trainer.actor.logit.grad is None
        assert all(p.grad is None for p in trainer.event_critic.parameters())
        assert any(p.grad is not None for p in trainer.latent_critic.parameters())
    assert torch.equal(cpu_before, torch.get_rng_state())
    assert numpy_before == trainer.rng.bit_generator.state
    assert not torch.equal(aux_before, trainer._critic_aux_torch_cpu)


def test_bounded_trainer_roundtrip_target_and_private_rng(tmp_path):
    from test_urban_dreamer_training import make_trainer
    template = make_trainer(tmp_path)
    config = replace(template.config, method=replace(template.config.method,
        critic_value_mode='bounded_cohort_v1', disagreement_weight=0.))
    def build():
        return MainlineTrainer(config=config, prepared_inputs=template.prepared_inputs, topology=template.topology)
    trainer = build()
    with torch.no_grad():
        next(trainer.target_latent_critic.parameters()).add_(.123)
    trainer._critic_aux_rng.random(3)
    state = copy.deepcopy(trainer.state_dict())
    restored = build()
    restored.load_state_dict(state)
    for first, second in zip(trainer.target_latent_critic.parameters(), restored.target_latent_critic.parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    np.testing.assert_array_equal(trainer._critic_aux_rng.random(10), restored._critic_aux_rng.random(10))
    assert torch.equal(trainer._critic_aux_torch_cpu, restored._critic_aux_torch_cpu)
    assert not restored.target_latent_critic.training
    legacy = template.state_dict()
    with pytest.raises(ValueError, match='contract mismatch'):
        restored.load_state_dict(legacy)
    with pytest.raises(ValueError, match='cannot load'):
        template.load_state_dict(state)


def test_parallel_worker_count_preserves_bounded_values_and_gradients(monkeypatch):
    from test_parallel_imagination import trainer_case, collect_result, close_parallel_imagination
    monkeypatch.setattr('urban_dreamer.learning.training.IMAGINED_HORIZON', 3)
    trainers = [bound(trainer_case(rollouts=2)), bound(trainer_case(rollouts=2))]
    try:
        results = [collect_result(trainer, count) for trainer, count in zip(trainers, (1, 2))]
        assert results[0][2] == results[1][2]
        for key in results[0][0]:
            torch.testing.assert_close(results[0][0][key], results[1][0][key], rtol=0, atol=0)
        for a, b in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert (results[0][0]['executed_q'] <= 0).all()
        assert (results[0][0]['executed_q'] >= -65 / 60 * .5).all()
    finally:
        for trainer in trainers:
            close_parallel_imagination(trainer)


def test_auxiliary_changes_only_value_loss_and_leaves_primary_actor_update_identical(tmp_path, monkeypatch):
    """Full pass isolates extra V data from actor/Q return and RNG statistics."""
    from test_urban_dreamer_training import make_trainer
    template = make_trainer(tmp_path)
    config = replace(template.config,
        method=replace(template.config.method, critic_value_mode='bounded_cohort_v1',
                       disagreement_weight=0., critic_terminal_aux_ratio=.25),
        resources=replace(template.config.resources, imagination_rollouts_per_pass=4))
    template = MainlineTrainer(config=config, prepared_inputs=template.prepared_inputs, topology=template.topology)
    monkeypatch.setattr('urban_dreamer.learning.fm_inference.configure_imagination_inference', lambda trainer: 'test')
    outcomes = []
    for ratio in (0., .25):
        trainer = copy.deepcopy(template)
        trainer.config = replace(config, method=replace(config.method, critic_terminal_aux_ratio=ratio))
        trainer._decoder_training_enabled = lambda: False
        trainer.online_replay = SimpleNamespace(has_active_seeds=True,
            sample_active_seeds=lambda **kwargs: None,
            sample_terminal_seeds=lambda **kwargs: SimpleNamespace(time=[0.]))
        calls = []
        def imagine(batch, index, *, critic_only=False, **kwargs):
            draw = torch.rand(())
            calls.append(critic_only)
            value_parameter = next(trainer.latent_critic.parameters()).reshape(-1)[0]
            loss = (value_parameter - (10. if critic_only else 1.)) ** 2
            if critic_only:
                return [], loss
            actor_parameter = next(trainer.actor.parameters()).reshape(-1)[0]
            q_parameter = next(trainer.event_critic.parameters()).reshape(-1)[0]
            row = dict(vehicle_id=['a'], executed_q=q_parameter[None], global_return=draw[None],
                global_advantage=(draw - .5)[None], entropy=(actor_parameter * 0 + .5)[None],
                log_prob=(actor_parameter * draw)[None], population=1,
                counterfactual_baseline=torch.zeros(1))
            return [row], loss
        trainer._imagine_from_seed = imagine
        torch.manual_seed(983)
        result = trainer._train_imagination_pass()
        outcomes.append((result, copy.deepcopy(trainer.actor.state_dict()),
                         copy.deepcopy(trainer.event_critic.state_dict()), torch.get_rng_state()))
        assert calls == [False] * 4 + ([True] if ratio else [])
    for key in ('actor_loss', 'event_critic_loss', 'actor_advantage_std', 'global_advantage_mean', 'policy_entropy_mean'):
        assert outcomes[0][0][key] == outcomes[1][0][key]
    assert outcomes[0][0]['latent_critic_loss'] != outcomes[1][0]['latent_critic_loss']
    for index in (1, 2):
        for key in outcomes[0][index]:
            torch.testing.assert_close(outcomes[0][index][key], outcomes[1][index][key], rtol=0, atol=0)
    assert torch.equal(outcomes[0][3], outcomes[1][3])


@pytest.mark.parametrize('target', [-1.01, .01, float('nan'), float('inf')])
def test_impossible_targets_fail_instead_of_clipping(target):
    trainer, _, _ = make_case()
    bound(trainer)
    with pytest.raises(ValueError, match='physical return bounds'):
        trainer._validate_critic_targets(torch.tensor([target]), [1.])
    trainer._validate_critic_targets(torch.tensor([-1.000001, .000001]), [1., 0.])


def test_tiny_budget_diagnostic_never_divides_tiny_or_zero_budgets():
    trainer, _, _ = make_case()
    bound(trainer)
    trainer._critic_diagnostics = dict(critic_predictions=0, critic_nonfinite_predictions=0,
        critic_bound_violations=0, critic_fraction_saturated=0, critic_terminal_predictions=0,
        critic_tiny_budget_predictions=0, critic_target_count=0, critic_target_bound_violations=0)
    result = trainer._value_prediction(torch.zeros(3, 2, 64), torch.zeros(3, 4), [0., 1e-40, 1.])
    assert torch.isfinite(result).all()
    assert trainer._critic_diagnostics['critic_tiny_budget_predictions'] == 1
    assert trainer._critic_diagnostics['critic_terminal_predictions'] == 1
    assert trainer._critic_diagnostics['critic_fraction_saturated'] == 0
    trainer._validate_critic_targets(result, [0., 1e-40, 1.])
    assert trainer._critic_diagnostics['critic_target_bound_violations'] == 0


def test_cuda_auxiliary_resume_rejects_missing_device_stream_before_loading(tmp_path, monkeypatch):
    from test_urban_dreamer_training import make_trainer
    template = make_trainer(tmp_path)
    config = replace(template.config, method=replace(template.config.method,
        critic_value_mode='bounded_cohort_v1', disagreement_weight=0.))
    trainer = MainlineTrainer(config=config, prepared_inputs=template.prepared_inputs, topology=template.topology)
    state = trainer.state_dict()
    state['rng']['critic_aux_torch_cuda'] = []
    trainer.device = torch.device('cuda:0')
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    with pytest.raises(ValueError, match='CUDA device count'):
        trainer.load_state_dict(state)
