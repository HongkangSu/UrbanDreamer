"""Component admission, paired initialization, and actual variant gradients."""
import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from tools import run_component_online_policy as runner
from test_cohort_targets import cohort_replay
from test_train_component_world_model import CAPACITIES, small_model
from test_urban_dreamer_training import make_trainer, mock_oracle_codec_gate_for_isolated_fm
from urban_dreamer.models.experimental_dynamics import apply_variant


def campaign(tmp_path):
    snapshot = tmp_path/'snapshot'
    (snapshot/'checkpoints').mkdir(parents=True)
    (snapshot/'run_config.toml').write_text('frozen config')
    (snapshot/'checkpoints/latest.pt').write_text('frozen codec parent')
    manifest = dict(entries=[dict(split='train', demand_group='a'), dict(split='validation', demand_group='b')])
    manifest['identity'] = runner._identity(manifest)
    (tmp_path/'development_manifest.json').write_text(json.dumps(manifest))
    parent = dict(config_sha256=runner.sha256(snapshot/'run_config.toml'),
        checkpoint_sha256=runner.sha256(snapshot/'checkpoints/latest.pt'),
        development_manifest_identity=manifest['identity'])
    (tmp_path/'protocol.json').write_text(json.dumps(parent))
    payload = dict(schema_version=1, variant='regression', seed=42, update=10,
        model={'dynamics._extra_state': dict(variant='regression', seed=42, version=1)},
        protocol=dict(parent=parent, variant=dict(variant='regression'), training_seed=42,
            mode='single_step', horizon=1, frozen_codec=True, planned_updates=20))
    path = tmp_path/'component.pt'
    torch.save(payload, path)
    return path, payload


def test_partial_offline_admission_is_explicit_and_verified(tmp_path):
    path, payload = campaign(tmp_path)
    actual, parent = runner.validate_component_source(tmp_path, path, runner.sha256(path), variant='regression', seed=42)
    assert actual == payload and parent == payload['protocol']['parent']
    with pytest.raises(ValueError, match='SHA256'):
        runner.validate_component_source(tmp_path, path, 'wrong', variant='regression', seed=42)
    with pytest.raises(ValueError, match='variant/seed'):
        runner.validate_component_source(tmp_path, path, runner.sha256(path), variant='gaussian', seed=42)
    payload['model']['dynamics._extra_state']['seed'] = 7
    torch.save(payload, path)
    with pytest.raises(ValueError, match='serialized variant/seed'):
        runner.validate_component_source(tmp_path, path, runner.sha256(path), variant='regression', seed=42)


def test_parent_and_config_tampering_are_rejected(tmp_path):
    path, payload = campaign(tmp_path)
    payload['protocol']['parent']['checkpoint_sha256'] = 'wrong'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='parent/schema'):
        runner.validate_component_source(tmp_path, path, runner.sha256(path), variant='regression', seed=42)
    (tmp_path/'snapshot/run_config.toml').write_text('changed')
    with pytest.raises(ValueError, match='snapshot digest'):
        runner.validate_component_source(tmp_path, path, runner.sha256(path), variant='regression', seed=42)


@pytest.mark.parametrize('variant', ['graph_fm', 'regression', 'gaussian'])
def test_online_variant_loss_trains_dynamics_and_observed_readout_without_codec(variant):
    model = apply_variant(small_model(), variant, seed=42)
    model.active_vehicle_fraction_readout.requires_grad_(True)
    trainer = SimpleNamespace(world_model=model, device=torch.device('cpu'),
        network=SimpleNamespace(capacities=CAPACITIES), metrics={},
        config=SimpleNamespace(method=SimpleNamespace(prior_loss_weight=.1)),
        validate_codec_ready=lambda: None)
    replay = cohort_replay()
    replay.sequence_length = 1
    batch = replay._batch_from_picks([(0, 0), (0, 1)])
    codec_before = runner.tensor_digest(model.flow_projector)
    dynamics = torch.optim.Adam(model.dynamics.parameters(), lr=1e-3)
    auxiliary = torch.optim.Adam(model.active_vehicle_fraction_readout.parameters(), lr=1e-3)
    dynamics_before = runner.tensor_digest(model.dynamics)
    readout_before = runner.tensor_digest(model.active_vehicle_fraction_readout)
    captured = []
    handle = model.active_vehicle_fraction_readout.register_forward_pre_hook(
        lambda m, args: captured.append(args[0].detach().clone()))
    loss, readout_loss = runner.ComponentPilotTrainer._world_loss(trainer, batch)
    handle.remove()
    assert torch.isfinite(loss) and torch.isfinite(readout_loss)
    loss.backward()
    readout_loss.backward()
    dynamics.step()
    auxiliary.step()
    assert runner.tensor_digest(model.dynamics) != dynamics_before
    assert runner.tensor_digest(model.active_vehicle_fraction_readout) != readout_before
    assert runner.tensor_digest(model.flow_projector) == codec_before
    assert all(p.grad is None for p in model.flow_projector.parameters())
    scoped = runner.prepare_transition_targets(batch, capacities=CAPACITIES)
    with torch.no_grad():
        observed = model.encode(scoped.transition_target_flow[:, 0]).unsqueeze(1)
    torch.testing.assert_close(captured[0], observed)


def component_trainer(tmp_path, variant):
    template = make_trainer(tmp_path/'template')
    config = replace(template.config, method=replace(template.config.method,
        critic_value_mode='bounded_cohort_v1', disagreement_weight=0.))
    trainer = runner.ComponentPilotTrainer(config=config, prepared_inputs=template.prepared_inputs,
        topology=template.topology, _offline_data=template.offline_data,
        _codec_validation_data=template.codec_validation_data, pilot_log=tmp_path/'pilot.jsonl')
    model = apply_variant(copy.deepcopy(trainer.world_model), variant, seed=42)
    payload = dict(variant=variant, seed=42, model=model.state_dict())
    return trainer, payload


def test_gaussian_adapter_preserves_fresh_policy_readout_and_rng(tmp_path):
    trainer, payload = component_trainer(tmp_path, 'gaussian')
    before = {name: runner.tensor_digest(getattr(trainer, name)) for name in
        ('actor', 'latent_critic', 'event_critic', 'target_latent_critic')}
    readout = runner.tensor_digest(trainer.world_model.active_vehicle_fraction_readout)
    for key, tensor in payload['model'].items():
        if key.startswith('active_vehicle_fraction_readout.'):
            tensor.fill_(12)
    rng = torch.get_rng_state().clone()
    trainer.initialize_component(payload, {'test': 'initialization'})
    assert torch.equal(torch.get_rng_state(), rng)
    assert runner.tensor_digest(trainer.world_model.active_vehicle_fraction_readout) == readout
    assert before == {name: runner.tensor_digest(getattr(trainer, name)) for name in before}
    assert {id(p) for group in trainer.optimizers['world_model_online'].param_groups for p in group['params']} == {
        id(p) for p in trainer.world_model.dynamics.parameters()}
    trainer.assert_frozen_codec()


def test_strict_resume_keeps_admission_target_rng_and_optimizer_state(tmp_path, monkeypatch):
    trainer, payload = component_trainer(tmp_path, 'regression')
    trainer.initialize_component(payload, {'source_sha': 'test'})
    # Oracle gate is isolated fixture plumbing; real smoke runs actual full validation.
    mock_oracle_codec_gate_for_isolated_fm(trainer, monkeypatch)
    for name in runner.pilot.MODEL_NAMES:
        model, optimizer = getattr(trainer, name), trainer.optimizers[name]
        sum(p.square().sum() for p in model.parameters()).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    state = copy.deepcopy(trainer.state_dict())
    torch.rand(2)
    with torch.no_grad():
        next(trainer.actor.parameters()).add_(1.)
    trainer.load_state_dict(state)
    assert torch.equal(torch.get_rng_state(), state['rng']['torch_cpu'])
    assert trainer.pilot_optimizer_steps == dict.fromkeys(runner.pilot.MODEL_NAMES, 1)
    for name in runner.pilot.MODEL_NAMES:
        assert len(trainer.optimizers[name].state) > 0
        for key, value in getattr(trainer, name).state_dict().items():
            torch.testing.assert_close(value, state['models'][name][key], rtol=0, atol=0)
    bad = copy.deepcopy(state)
    bad['metrics']['component_admission']['source_sha'] = 'changed'
    with pytest.raises(ValueError, match='admission'):
        trainer.load_state_dict(bad)
    assert set(state) == {'models', 'optimizers', 'replay', 'rng', 'metrics', 'next_environment_episode', 'reward_contract'}


def test_physical_reward_path_remains_production_explicit_vehicle_accounting():
    from urban_dreamer.learning.training import MainlineTrainer
    assert runner.ComponentPilotTrainer._imagine_from_seed is MainlineTrainer._imagine_from_seed
    assert runner.ComponentPilotTrainer._collect_episode is MainlineTrainer._collect_episode
