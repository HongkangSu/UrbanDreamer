"""Training-only scale provenance and exact small CPU residual continuation."""
import copy

import numpy as np
import pytest
import torch

from tools import residual_experiment_data as data
from test_cohort_targets import cohort_replay
from test_train_component_world_model import small_model, CAPACITIES
from urban_dreamer.models.experimental_dynamics import apply_variant


class Source:
    sequence_length = 1

    def __init__(self):
        self.replay = cohort_replay()
        self.replay.sequence_length = 1
        self.manifest = dict(identity='training-fixture', entries=[
            dict(split='train', windows=2, demand_group='train-only', path='fixture')])

    def batch_from_starts(self, episode, starts, **kwargs):
        return self.replay._batch_from_picks([(0, int(start)) for start in starts])


def center():
    model = apply_variant(small_model(), 'regression', seed=42).requires_grad_(False).eval()
    sampler = np.random.default_rng(99)
    for _ in range(2):
        sampler.choice(1, p=[1.])
        sampler.integers(2, size=2)
    provenance = dict(mean_source_sha256='a'*64,
        codec_sha256=data.state_digest(model.flow_projector.state_dict()), manifest_sha256='c'*64,
        mean_seed=42, mean_updates=2, mean_exposures=4, mean_training_seconds=1.,
        mean_data_rng=copy.deepcopy(sampler.bit_generator.state),
        mean_data_rng_identity=data._identity(sampler.bit_generator.state),
        mean_training_protocol=dict(training_manifest_identity='training-fixture',
            data_seed=99, batch_size=2, sampler='uniform_legal_windows_with_replacement_episode_batches'))
    return model, provenance


def test_scale_is_reproducible_training_only_and_preserves_rng_weights():
    model, provenance = center()
    source = Source()
    torch.manual_seed(192)
    rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    before = data.state_digest(model.state_dict())
    first = data.calibrate_scale(model, source, CAPACITIES, provenance, count=5, batch_size=2)
    second = data.calibrate_scale(model, source, CAPACITIES, provenance, count=5, batch_size=2)
    assert first == second and len(first['samples']) == 5
    assert torch.equal(rng, torch.get_rng_state())
    np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
    assert data.state_digest(model.state_dict()) == before
    assert data.validate_scale(first, source=source, provenance=provenance).shape == (64,)
    assert {row['demand_group'] for row in first['samples']} == {'train-only'}
    source.manifest['entries'][0]['split'] = 'validation'
    with pytest.raises(ValueError, match='training partition'):
        data.calibrate_scale(model, source, CAPACITIES, provenance, count=2)


def test_scale_tampering_and_sample_identity_fail_even_if_rehashed():
    model, provenance = center()
    source = Source()
    artifact = data.calibrate_scale(model, source, CAPACITIES, provenance, count=2)
    bad = copy.deepcopy(artifact)
    bad['scale'][0] *= 2
    with pytest.raises(ValueError, match='identity'):
        data.validate_scale(bad)
    bad['identity'] = data._identity({k:v for k,v in bad.items() if k != 'identity'})
    with pytest.raises(ValueError, match='RMS/floor'):
        data.validate_scale(bad)
    bad = copy.deepcopy(artifact)
    bad['samples'][0]['demand_group'] = 'held-out'
    bad['sample_identity'] = data._identity(bad['samples'])
    bad['identity'] = data._identity({k:v for k,v in bad.items() if k != 'identity'})
    with pytest.raises(ValueError, match='outside training'):
        data.validate_scale(bad, source=source)


def test_scale_json_sha_matches_actual_artifact(tmp_path):
    from tools.component_experiment_data import atomic_json
    model, provenance = center()
    artifact = data.calibrate_scale(model, Source(), CAPACITIES, provenance, count=2)
    path = tmp_path/'scale.json'
    atomic_json(path, artifact)
    assert data.json_sha256(artifact) == data.sha256(path)


def test_residual_small_cpu_stage_and_exact_next_update_resume(tmp_path):
    from tools import train_residual_world_model as trainer
    model, provenance = center()
    source = Source()
    artifact = data.calibrate_scale(model, source, CAPACITIES, provenance, count=4, batch_size=2)
    data.install_residual(model, artifact, initialization_seed=71)
    protocol = trainer.stage_protocol(model, source, {}, provenance, artifact,
        seed=71, data_seed=99, updates=3, batch_size=2, precision='fp32', learning_rate=1e-4)
    rows = trainer.run_stage(model, source, CAPACITIES, tmp_path/'first', protocol, artifact,
        stop_after=2, checkpoint_every=1)
    checkpoint = trainer.read_stage_checkpoint(tmp_path/'first')
    control_rng = np.random.default_rng()
    control_rng.bit_generator.state = copy.deepcopy(provenance['mean_data_rng'])
    assert rows[0]['episode'] == int(control_rng.choice(1, p=[1.]))
    assert rows[0]['starts'] == control_rng.integers(2, size=2).tolist()
    assert protocol['training_seed'] == protocol['initialization_seed'] == 71
    # A crash before publishing the latest alias must not strand a committed generation.
    alias = tmp_path/'first/latest.pt'
    alias.unlink()
    assert trainer.read_stage_checkpoint(tmp_path/'first')['update'] == 2
    assert checkpoint['update'] == 2 and rows[-1]['total_optimization_updates'] == 4
    assert checkpoint['exposures'] == 4 and checkpoint['optimizer']['state']
    assert all(not p.requires_grad for p in model.dynamics.mean_dynamics.parameters())
    expected = trainer.run_stage(model, source, CAPACITIES, tmp_path/'first', protocol, artifact,
        resume=copy.deepcopy(checkpoint), checkpoint_every=1)
    expected_state = data.state_digest(model.state_dict())
    from tools.evaluate_component_world_models import restore_component_checkpoint
    from urban_dreamer.models.experimental_dynamics import variant_metadata
    shell = small_model()
    restore_component_checkpoint(shell, checkpoint, {})
    assert variant_metadata(shell)['variant'] == data.VARIANT
    assert variant_metadata(shell)['noise_space'].startswith('full normalized')
    torch.rand(9)
    actual = trainer.run_stage(shell, source, CAPACITIES, tmp_path/'restored', protocol, artifact,
        resume=checkpoint, checkpoint_every=1)
    assert expected_state == data.state_digest(shell.state_dict())
    for key in ('loss', 'gradient_norm', 'episode', 'starts', 'exposures'):
        assert expected[-1][key] == actual[-1][key]
    altered = copy.deepcopy(checkpoint)
    altered['protocol']['scale_artifact_sha256'] = 'd'*64
    with pytest.raises(ValueError, match='identity'):
        data.restore_residual_model(small_model(), altered, {})
    bad_protocol = dict(protocol, data_seed=100)
    with pytest.raises(ValueError, match='protocol'):
        trainer.run_stage(shell, source, CAPACITIES, tmp_path/'bad', bad_protocol, artifact,
            resume=checkpoint)


def test_evaluator_restored_residual_uses_stochastic_transitions_and_recursion(tmp_path):
    from tools import train_residual_world_model as trainer
    from tools.evaluate_component_world_models import restore_component_checkpoint, sample_trajectories
    model, provenance = center()
    source = Source()
    artifact = data.calibrate_scale(model, source, CAPACITIES, provenance, count=2)
    data.install_residual(model, artifact, initialization_seed=12)
    protocol = trainer.stage_protocol(model, source, {}, provenance, artifact,
        seed=12, data_seed=99, updates=1, batch_size=2, precision='fp32', learning_rate=1e-4)
    trainer.run_stage(model, source, CAPACITIES, tmp_path, protocol, artifact)
    checkpoint = trainer.read_stage_checkpoint(tmp_path)
    restored = restore_component_checkpoint(small_model(), checkpoint, {}).eval()
    batch = cohort_replay()._batch_from_picks([(0, 0)])
    with torch.no_grad():
        z = restored.encode(batch.flow[:, 0])
        kwargs = dict(action=batch.action, events=batch.action_events, horizons=[1, 2],
            samples=4, sample_chunk=2, steps=2, capacities=torch.tensor(CAPACITIES),
            limits=torch.ones(3)*10, seed=19)
        draws, variance = sample_trajectories(restored, z, **kwargs)
        assert draws[1].shape[:2] == (4, 1) and draws[2].shape[:2] == (4, 1)
        assert variance[1][0] > 0 and variance[2][0] > 0
        repeated, _ = sample_trajectories(restored, z, **kwargs)
        torch.testing.assert_close(draws[2], repeated[2], rtol=0, atol=0)
        unchunked, _ = sample_trajectories(restored, z, **dict(kwargs, sample_chunk=4))
        singles, _ = sample_trajectories(restored, z, **dict(kwargs, sample_chunk=1))
        torch.testing.assert_close(unchunked[2], singles[2], rtol=2e-5, atol=2e-5)
        original_transition = restored.transition
        noise_records = []
        def capture_transition(*args, **options):
            noise_records.append(options['noise'].clone())
            return original_transition(*args, **options)
        restored.transition = capture_transition
        sample_trajectories(restored, z, **dict(kwargs, sample_chunk=1))
        single_noise = [torch.cat(noise_records[depth::2]) for depth in range(2)]
        noise_records.clear()
        sample_trajectories(restored, z, **dict(kwargs, sample_chunk=4))
        for depth in range(2):
            torch.testing.assert_close(single_noise[depth], noise_records[depth], rtol=0, atol=0)
        restored.transition = original_transition
        from tools.evaluate_component_world_models import evaluate_condition
        rows = evaluate_condition(restored, batch, torch.tensor(CAPACITIES), torch.ones(3)*10,
            horizons=[1,2], samples=4, sample_chunk=2, steps=2, device='cpu',
            scales=np.ones(2), metadata={}, seed=19)
        centered = [row for row in rows if row['predictor'] == 'center_only']
        assert len(centered) == 3 and all(row['draws'] == 1 for row in centered)
        # Compare own center trajectory to the same evaluator using mean dynamics only.
        reference = copy.deepcopy(restored)
        reference.dynamics = reference.dynamics.mean_dynamics
        reference_rows = evaluate_condition(reference, batch, torch.tensor(CAPACITIES), torch.ones(3)*10,
            horizons=[1,2], samples=4, sample_chunk=2, steps=2, device='cpu',
            scales=np.ones(2), metadata={}, seed=19)
        deterministic = [row for row in reference_rows if row['predictor'] == 'predictor']
        for row, expected in zip(centered, deterministic):
            assert row['protocol'] == expected['protocol'] and row['horizon'] == expected['horizon']
            assert row['count_mae_all'] == pytest.approx(expected['count_mae_all'], abs=1e-7)


def test_checkpoint_seal_rejects_changed_file(tmp_path):
    from tools import train_residual_world_model as trainer
    from tools.component_experiment_data import atomic_json
    path = tmp_path/'latest.pt'
    path.write_bytes(b'original')
    atomic_json(tmp_path/'checkpoint_identity.json', dict(update=1, checkpoint_sha256=data.sha256(path)))
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA256'):
        trainer.read_stage_checkpoint(tmp_path)


def test_source_identity_codec_and_completed_budget_checks(tmp_path, monkeypatch):
    from tools.component_experiment_data import atomic_json
    snapshot = tmp_path/'snapshot'
    (snapshot/'checkpoints').mkdir(parents=True)
    (snapshot/'run_config.toml').write_text('config')
    (snapshot/'checkpoints/latest.pt').write_text('codec parent')
    manifest = dict(entries=[dict(split='train', demand_group='a'),
                             dict(split='validation', demand_group='b')])
    manifest['identity'] = data._identity(manifest)
    atomic_json(tmp_path/'development_manifest.json', manifest)
    parent = dict(config_sha256=data.sha256(snapshot/'run_config.toml'),
        checkpoint_sha256=data.sha256(snapshot/'checkpoints/latest.pt'),
        development_manifest_identity=manifest['identity'])
    atomic_json(tmp_path/'protocol.json', parent)
    model, _ = center()
    payload = dict(schema_version=1, variant='regression', seed=42, update=2, exposures=4,
        data_rng=np.random.default_rng(5).bit_generator.state, torch_rng=torch.get_rng_state(),
        cuda_rng=None, optimizer={},
        training_seconds=1., model=model.state_dict(), protocol=dict(parent=parent,
            mode='single_step', horizon=1, frozen_codec=True, training_seed=42,
            planned_updates=2, batch_size=2, data_seed=5,
            sampler='uniform_legal_windows_with_replacement_episode_batches', variant={'variant': 'regression'}))
    path = tmp_path/'mean.pt'
    torch.save(payload, path)
    monkeypatch.setattr(data, 'load_frozen', lambda *a, **kw: (None,None,None,small_model(),parent))
    with pytest.raises(ValueError, match='SHA256'):
        data.load_center(tmp_path, path, 'wrong')
    _, _, _, imported, _, provenance = data.load_center(tmp_path, path, data.sha256(path))
    assert imported.dynamics.variant == 'regression' and provenance['mean_updates'] == 2
    assert all(not p.requires_grad for p in imported.parameters())
    payload['update'] = 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match='completed'):
        data.load_center(tmp_path, path, data.sha256(path))
    payload['update'] = 2
    rng_state = payload.pop('data_rng')
    torch.save(payload, path)
    with pytest.raises(ValueError, match='full RNG'):
        data.load_center(tmp_path, path, data.sha256(path))
    payload['data_rng'] = rng_state
    payload['model']['flow_projector.input_proj.weight'].add_(1.)
    torch.save(payload, path)
    with pytest.raises(ValueError, match='codec differs'):
        data.load_center(tmp_path, path, data.sha256(path))


def test_staged_summary_is_explicit_and_ordinary_pairing_remains_rejected(tmp_path):
    from test_component_prediction_summary import evaluation
    from tools.component_experiment_data import atomic_json
    from tools.summarize_component_predictions import read_evaluation, create_report, validate_pairing
    residual = evaluation(data.VARIANT)
    protocol = residual['protocol']
    protocol.update(predictor_set=['predictor','persistence','endpoint_codec','center_only'],
        sampling_protocol='residual_per_trajectory_depth_noise_v1', training_update=2048,
        training_protocol=dict(kind='frozen_regression_residual_cfm_v1', frozen_codec=True,
            training_seed=42, parent=protocol['campaign']),
        staged_accounting=dict(kind='frozen_regression_plus_residual_stage_v1',
            center_updates=2048, stage_updates=2048, total_updates=4096,
            center_exposures=32768, stage_exposures=32768, total_exposures=65536,
            calibration_forward_exposures=4096))
    for state in residual['states'].values():
        state['center_only'] = dict(state['persistence'], predictor='center_only')
    atomic_json(tmp_path/'protocol.json', protocol)
    atomic_json(tmp_path/'conditions.json', residual['conditions'])
    atomic_json(tmp_path/'summary.json', dict(status='complete', conditions=2))
    import json
    (tmp_path/'states.jsonl').write_text(''.join(json.dumps(row)+'\n'
        for state in residual['states'].values() for row in state.values()))
    loaded = read_evaluation(tmp_path)
    report = create_report([loaded], repetitions=10)
    assert loaded['training_seed'] == 42
    assert {row['reference'] for row in report['paired_differences']} == {'persistence','center_only'}
    ordinary = evaluation('regression')
    for pair in ([loaded,ordinary], [ordinary,loaded]):
        with pytest.raises(ValueError, match='incompatible'):
            validate_pairing(pair)
