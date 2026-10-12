"""Component admission, paired initialization, and actual variant gradients."""
import copy
from dataclasses import replace
import json
import random
from types import SimpleNamespace

import numpy as np
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


def component_trainer(tmp_path, variant, *, policy_seed=42):
    template = make_trainer(tmp_path/'template')
    config = replace(template.config, run=replace(template.config.run, seed=policy_seed),
        simulation=replace(template.config.simulation, demand_duration_seconds=30.),
        method=replace(template.config.method,
        critic_value_mode='bounded_cohort_v1', disagreement_weight=0.))
    config = replace(config, semantic_fingerprint=runner._semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
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


@pytest.mark.parametrize('overrides,expected', [({}, (42, 42)),
    ({'seed': 7}, (7, 7)), ({'policy_seed': 43}, (42, 43)),
    ({'component_seed': 11}, (11, 42)),
    ({'seed': 9, 'component_seed': 42}, (42, 9)),
    ({'seed': 9, 'component_seed': 42, 'policy_seed': 0}, (42, 0))])
def test_legacy_seed_defaults_and_explicit_overrides(overrides, expected):
    args = SimpleNamespace(**dict(dict(seed=42), **overrides))
    assert runner.resolved_seeds(args) == expected
    protocol = runner.seed_protocol(args)
    if set(overrides) <= {'seed'}:
        assert protocol is None
    else:
        assert protocol['schema_version'] == 1
        assert (protocol['component_seed'], protocol['policy_seed']) == expected
        assert protocol['online_demand_seed'] == expected[1]


def test_fixed_component_has_reproducible_independent_policy_and_rng_initialization(tmp_path):
    _, payload = component_trainer(tmp_path/'source', 'gaussian')
    results = []
    for index, seed in enumerate((43, 44, 43)):
        trainer, _ = component_trainer(tmp_path/str(index), 'gaussian', policy_seed=seed)
        # Payload creation consumes adapter RNG; capture the state immediately
        # before import to prove import itself leaves the learner RNG unchanged.
        state = torch.get_rng_state().clone()
        numpy_state = copy.deepcopy(trainer.rng.bit_generator.state)
        trainer.initialize_component(payload, dict(component_seed=42, policy_seed=seed))
        assert torch.equal(torch.get_rng_state(), state)
        assert trainer.rng.bit_generator.state == numpy_state
        results.append(dict(policy={name: runner.tensor_digest(getattr(trainer, name)) for name in
            (*runner.pilot.MODEL_NAMES, 'target_latent_critic')},
            readout=runner.tensor_digest(trainer.world_model.active_vehicle_fraction_readout),
            dynamics=runner.tensor_digest(trainer.world_model.dynamics),
            codec=runner.tensor_digest(trainer.world_model.flow_projector),
            torch_draw=torch.rand(4).tolist(), numpy_draw=trainer.rng.random(4).tolist(),
            global_numpy_draw=np.random.random(4).tolist(), python_draw=[random.random() for _ in range(4)],
            auxiliary_draw=trainer._critic_aux_rng.random(4).tolist()))
    assert results[0] == results[2]
    assert results[0]['dynamics'] == results[1]['dynamics']
    assert results[0]['codec'] == results[1]['codec']
    for key in ('readout', 'torch_draw', 'numpy_draw', 'global_numpy_draw', 'python_draw', 'auxiliary_draw'):
        assert results[0][key] != results[1][key]
    for name in results[0]['policy']:
        assert results[0]['policy'][name] != results[1]['policy'][name]


def test_cli_emits_legacy_or_versioned_seed_identity_before_any_collection(tmp_path, monkeypatch):
    template, payload = component_trainer(tmp_path/'fixture', 'regression')
    payload.update(update=10, protocol=dict(planned_updates=20,
        training_manifest_identity=template.offline_data.manifest['identity']))
    source_config = template.config
    calls = []
    def load_source(*args, **kwargs):
        calls.append(kwargs['seed'])
        assert kwargs['seed'] == payload['seed']
        return payload, {'parent': 'fixed'}
    monkeypatch.setattr(runner, 'validate_component_source', load_source)
    monkeypatch.setattr(runner, 'load_config', lambda _: source_config)
    monkeypatch.setattr(runner.pilot, 'validate_prepared_inputs', lambda _: template.prepared_inputs)
    monkeypatch.setattr(runner.pilot, 'load_traffic_topology', lambda _: template.topology)
    monkeypatch.setattr(runner, 'dataset', lambda *args, **kwargs: template.offline_data)
    captured = []
    class Captured(RuntimeError):
        pass
    def stop_before_collection(trainer, **kwargs):
        captured.append((trainer.config, copy.deepcopy(kwargs['protocol']),
                         copy.deepcopy(trainer.metrics['component_admission'])))
        raise Captured
    monkeypatch.setattr(runner.pilot, 'OnlinePilot', stop_before_collection)
    base = ['--experiment-dir', str(tmp_path), '--checkpoint', str(tmp_path/'component.pt'),
        '--checkpoint-sha256', 'fixed-sha', '--variant', 'regression', '--resume']
    for index, flags in enumerate(([], ['--seed', '42'], ['--component-seed', '42', '--policy-seed', '43'],
                                  ['--component-seed', '42', '--policy-seed', '42'])):
        with pytest.raises(Captured):
            runner.main([*base, '--output-dir', str(tmp_path/f'run{index}'), *flags])
    legacy_config, legacy, legacy_admission = captured[0]
    assert legacy == captured[1][1]
    assert legacy['schema_version'] == 1 and 'seeds' not in legacy
    assert 'seeds' not in legacy_admission and legacy_admission['seed'] == 42
    assert calls == [42, 42, 42, 42]
    config, split, admission = captured[2]
    assert config.run.seed == 43 and split['schema_version'] == 2
    assert split['seeds']['component_seed'] == admission['seed'] == 42
    assert split['seeds']['policy_seed'] == 43
    assert admission['seeds'] == split['seeds']
    assert config.method.evaluation_demand_seed == legacy_config.method.evaluation_demand_seed
    assert config.semantic_fingerprint != legacy_config.semantic_fingerprint
    assert captured[3][0].semantic_fingerprint == legacy_config.semantic_fingerprint
    # Equal numerical seeds still explicitly opt into the new protocol version.
    assert captured[3][1]['schema_version'] == 2
    for key in set(legacy) - {'schema_version', 'source', 'runtime_config_fingerprint'}:
        assert legacy[key] == split[key]


@pytest.mark.parametrize('version', [1, 2])
def test_seed_checkpoint_resume_and_changed_identity_rejection(tmp_path, monkeypatch, version):
    policy_seed = 42 if version == 1 else 43
    trainer, payload = component_trainer(tmp_path/'first', 'regression', policy_seed=policy_seed)
    admission = dict(seed=42, source_sha='fixed')
    if version == 2:
        admission['seeds'] = runner.seed_protocol(SimpleNamespace(seed=42, component_seed=42, policy_seed=43))
    trainer.initialize_component(payload, admission)
    mock_oracle_codec_gate_for_isolated_fm(trainer, monkeypatch)
    for name in runner.pilot.MODEL_NAMES:
        sum(p.square().sum() for p in getattr(trainer, name).parameters()).backward()
        trainer.optimizers[name].step()
        trainer.optimizers[name].zero_grad(set_to_none=True)
    protocol = dict(schema_version=version, source=admission, runner_sha256=runner.sha256(runner.__file__))
    if version == 2:
        protocol['seeds'] = admission['seeds']
    output = tmp_path/'saved'
    first = runner.pilot.OnlinePilot(trainer, output=output, input_hash='fixed', protocol=protocol, online_episodes=1)
    first.resume()
    checkpoint_before = runner.sha256(output/'checkpoints/latest.pt')
    restored, _ = component_trainer(tmp_path/'second', 'regression', policy_seed=policy_seed)
    restored.offline_data = trainer.offline_data
    restored.initialize_component(payload, admission)
    mock_oracle_codec_gate_for_isolated_fm(restored, monkeypatch)
    second = runner.pilot.OnlinePilot(restored, output=output, input_hash='fixed', protocol=protocol, online_episodes=1)
    second.resume()
    assert runner.strict_resume_audit(restored, output, 'fixed')['exact_state_restored']
    assert runner.sha256(output/'checkpoints/latest.pt') == checkpoint_before
    assert restored.pilot_optimizer_steps == dict.fromkeys(runner.pilot.MODEL_NAMES, 1)
    fields = [('source', 'seed'), (None, 'runner_sha256'), (None, 'schema_version')]
    if version == 2:
        fields.extend(('seeds', field) for field in ('component_seed', 'policy_seed', 'online_demand_seed'))
    for section, field in fields:
        changed = copy.deepcopy(protocol)
        target = changed[section] if section else changed
        target[field] = 'changed' if isinstance(target[field], str) else target[field] + 1
        with pytest.raises(ValueError, match='protocol/source identity'):
            runner.pilot.OnlinePilot(restored, output=output, input_hash='fixed',
                protocol=changed, online_episodes=1).resume()
    state = copy.deepcopy(restored.state_dict())
    if version == 2:
        state['metrics']['component_admission']['seeds']['policy_seed'] = 44
    else:
        state['metrics']['component_admission']['seed'] = 44
    with pytest.raises(ValueError, match='admission'):
        restored.load_state_dict(state)


@pytest.mark.parametrize('mode', ['independent_cohort', 'recursive_fm', 'sample_ms', 'mean_ms'])
def test_other_completed_supervision_modes_remain_explicitly_unadmitted(tmp_path, mode):
    path, payload = campaign(tmp_path)
    payload.update(variant='graph_fm', model={})
    payload['protocol'].update(mode=mode, horizon=3, variant=dict(variant='graph_fm'))
    payload['update'] = payload['protocol']['planned_updates']
    torch.save(payload, path)
    with pytest.raises(ValueError, match='single-step checkpoint'):
        runner.validate_component_source(tmp_path, path, runner.sha256(path), variant='graph_fm', seed=42)
