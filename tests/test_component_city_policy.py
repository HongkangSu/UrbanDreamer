"""City opt-in admission and genuine serial trainer executor lifecycle."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import run_component_online_policy as runner
from urban_dreamer.learning import training
from test_city_imagination import table
from test_event_imagination_protocol import network
from test_movement_controls import fixture
from test_urban_dreamer_event_returns import make_case


def admission_case(tmp_path, **xml_kwargs):
    path = fixture(tmp_path, **xml_kwargs)
    prepared = SimpleNamespace(network_xml=path, input_hashes={'network_xml': runner.sha256(path)})
    topology = SimpleNamespace(network=SimpleNamespace(edge_ids=('a', 'b', 'c', 'd'),
        adjacency_list=((2,), (3,), (), ())))
    simulation = SimpleNamespace(traffic_model='meso', meso_junction_control=True,
        meso_lane_queue=False, meso_multi_queue=False, session_protocol='event',
        observation_step_seconds=1., sumo_extra_args=('--collision.action', 'warn'))
    args = SimpleNamespace(city_queue_order='strict', city_discharge_headway=1.,
        city_receiving_headway=1., city_conflict_clearance=1.)
    return SimpleNamespace(simulation=simulation), prepared, topology, args


def test_city_control_provenance_hashes_network_table_and_sources(tmp_path):
    config, prepared, topology, args = admission_case(tmp_path)
    controls, preflight, protocol = runner.city_executor_admission(config, prepared, topology, args)
    assert protocol['network_sha256'] == runner.sha256(prepared.network_xml)
    assert protocol['control_support'] == {'exact_control': 2}
    assert protocol['movement_count'] == len(controls.movements) == 2
    assert not protocol['parameters']['uncertain_order']
    assert 'spatial initial order' in protocol['approximation']
    assert len(protocol['source_sha256']) == 4
    args.city_queue_order = 'seeded_uncertain'
    other = runner.city_executor_admission(config, prepared, topology, args)[2]
    assert other['parameters']['uncertain_order']
    assert other['control_table_sha256'] == protocol['control_table_sha256']
    prepared.network_xml.write_text(prepared.network_xml.read_text().replace('offset="1"', 'offset="2"'))
    with pytest.raises(ValueError, match='digest'):
        runner.city_executor_admission(config, prepared, topology, args)
    prepared.input_hashes['network_xml'] = runner.sha256(prepared.network_xml)
    changed = runner.city_executor_admission(config, prepared, topology, args)[2]
    assert changed['control_table_sha256'] != protocol['control_table_sha256']


@pytest.mark.parametrize('field,value', [('traffic_model', 'micro'), ('meso_junction_control', False),
    ('meso_lane_queue', True), ('meso_multi_queue', True), ('observation_step_seconds', .5),
    ('sumo_extra_args', ('--additional-files', 'other.xml')),
    ('sumo_extra_args', ('--tls.all-off', 'true'))])
def test_city_rejects_unverified_physics_and_overrides(tmp_path, field, value):
    config, prepared, topology, args = admission_case(tmp_path)
    setattr(config.simulation, field, value)
    with pytest.raises(ValueError):
        runner.city_executor_admission(config, prepared, topology, args)


@pytest.mark.parametrize('xml_kwargs', [dict(program=False), dict(junction_type='rail_crossing')])
def test_city_rejects_unsupported_controls_and_rail_approximation(tmp_path, xml_kwargs):
    config, prepared, topology, args = admission_case(tmp_path, **xml_kwargs)
    with pytest.raises(ValueError, match='unsupported/rail'):
        runner.city_executor_admission(config, prepared, topology, args)


def city_trainer():
    trainer, batch, _ = make_case(deadline=60., lengths=(1000., 1000., 1000.))
    trainer.__class__ = runner.ComponentPilotTrainer
    trainer.metrics = {}
    trainer.config.resources = SimpleNamespace(world_model_imagination_compile=False,
        world_model_imagination_precision='fp32')
    trainer.config.simulation.traffic_model = 'meso'
    trainer.candidates.traffic_model = 'meso'
    trainer.network = trainer.candidates.network = network((1000., 1000., 1000., 1000.),
        graph=((1, 3), (2, 3), (), (2,)))
    trainer.configure_city_executor(table(trainer.network), dict(parameters=dict(uncertain_order=False,
        discharge_headway_seconds=1., receiving_headway_seconds=1., conflict_clearance_seconds=1.)))
    return trainer, batch


def test_arbiter_seed_hash_does_not_advance_primary_rng_and_auxiliary_is_separate(monkeypatch):
    monkeypatch.setenv('URB_IMAGINATION_WORKERS', '0')
    trainer, _ = city_trainer()
    before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    first = trainer._create_imagination_executor(rollout_index=0, critic_only=False)
    repeat = trainer._create_imagination_executor(rollout_index=0, critic_only=False)
    second = trainer._create_imagination_executor(rollout_index=1, critic_only=False)
    auxiliary = trainer._create_imagination_executor(rollout_index=0, critic_only=True)
    assert first is not repeat and first.seed == repeat.seed
    assert len({first.seed, second.seed, auxiliary.seed}) == 3
    assert torch.equal(torch.get_rng_state(), before)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
    first.source_ready[5] = 90.
    assert not repeat.source_ready and not second.source_ready


def test_real_rollout_keeps_one_calendar_across_windows_and_discards_it_after(monkeypatch):
    monkeypatch.setenv('URB_IMAGINATION_WORKERS', '0')
    trainer, batch = city_trainer()
    created, windows = [], []
    create, advance = trainer._create_imagination_executor, trainer._advance_imagination_window
    def factory(**kwargs):
        result = create(**kwargs)
        created.append(result)
        return result
    def window(executor, states, **kwargs):
        windows.append((executor, kwargs['start_time']))
        return advance(executor, states, **kwargs)
    trainer._create_imagination_executor = factory
    trainer._advance_imagination_window = window
    original = deepcopy(batch.vehicles)
    for index in range(2):
        rows, loss = trainer._imagine_from_seed(batch, 0, rollout_index=index)
        assert rows and torch.isfinite(loss)
    assert len(created) == 2 and created[0] is not created[1]
    assert [start for _, start in windows] == [0., 30., 0., 30.]
    assert [executor for executor, _ in windows] == [created[0], created[0], created[1], created[1]]
    assert created[0].end_time == created[1].end_time == 60.
    assert batch.vehicles == original
    assert trainer.metrics['city_executor_rollouts'] == 2
    assert trainer.metrics['city_executor_windows'] == 4


def test_default_trainer_matches_legacy_direct_function_without_rng_shift():
    trainer, batch, _ = make_case(deadline=60.)
    state = torch.get_rng_state().clone()
    assert trainer._create_imagination_executor(rollout_index=0, critic_only=False) is None
    assert torch.equal(torch.get_rng_state(), state)
    rows, loss = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    torch.set_rng_state(state)
    trainer._advance_imagination_window = lambda executor, states, **kwargs: training.advance_imagined_window(states, **kwargs)
    expected, expected_loss = trainer._imagine_from_seed(batch, 0, rollout_index=0)
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
    for actual, old in zip(rows, expected, strict=True):
        assert actual.keys() == old.keys()
        for key in actual:
            if torch.is_tensor(actual[key]):
                torch.testing.assert_close(actual[key], old[key], rtol=0, atol=0)
            else:
                assert actual[key] == old[key]


@pytest.mark.parametrize('kind', ['parallel', 'compiled', 'bf16'])
def test_city_never_silently_falls_back_to_parallel_or_compiled_legacy(monkeypatch, kind):
    monkeypatch.setenv('URB_IMAGINATION_WORKERS', '1' if kind == 'parallel' else '0')
    trainer, _ = city_trainer()
    if kind == 'compiled':
        trainer.config.resources.world_model_imagination_compile = True
    if kind == 'bf16':
        trainer.config.resources.world_model_imagination_precision = 'bf16'
    with pytest.raises(ValueError, match='serial FP32'):
        trainer._train_imagination_pass()


def test_city_checkpoint_admission_rejects_changed_controls_and_parameters(tmp_path, monkeypatch):
    from test_component_online_policy import component_trainer
    from test_urban_dreamer_training import mock_oracle_codec_gate_for_isolated_fm
    config, prepared, topology, args = admission_case(tmp_path)
    controls, _, protocol = runner.city_executor_admission(config, prepared, topology, args)
    trainer, payload = component_trainer(tmp_path/'learner', 'regression')
    admission = dict(source_sha='fixed', imagination_executor=protocol)
    with pytest.raises(ValueError, match='configured city executor'):
        trainer.initialize_component(payload, admission)
    trainer.configure_city_executor(controls, protocol)
    trainer.initialize_component(payload, admission)
    mock_oracle_codec_gate_for_isolated_fm(trainer, monkeypatch)
    original = deepcopy(trainer.state_dict())
    trainer.load_state_dict(original)
    for field in ('control_table_sha256', 'network_sha256', 'queue_order'):
        changed = deepcopy(original)
        changed['metrics']['component_admission']['imagination_executor'][field] = 'changed'
        with pytest.raises(ValueError, match='admission'):
            trainer.load_state_dict(changed)
    changed = deepcopy(original)
    changed['metrics']['component_admission']['imagination_executor']['parameters']['discharge_headway_seconds'] = 2.
    with pytest.raises(ValueError, match='admission'):
        trainer.load_state_dict(changed)
