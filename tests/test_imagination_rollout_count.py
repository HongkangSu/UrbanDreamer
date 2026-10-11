"""Configurable rollout batches preserve coverage and independent random streams."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_urban_dreamer_config import _write_config
from test_urban_dreamer_world_model import _small_dynamics
from test_parallel_imagination import trainer_case, collect_result
from urban_dreamer.learning.parallel_imagination import close_parallel_imagination
from urban_dreamer.learning.fm_inference import configure_imagination_inference, _VelocityInference, ImaginationInferenceProfile
from urban_dreamer.learning.training import resolved_imagination_budget
from urban_dreamer.models.world_model import DynamicsOutput
from urban_dreamer.workflow.config import load_config, ConfigError


def test_default_eight_preserves_legacy_fingerprint_and_sixteen_is_fingerprinted(tmp_path):
    legacy = load_config(_write_config(tmp_path / 'old.toml'))
    explicit = load_config(_write_config(tmp_path / 'explicit.toml', extra='\nimagination_rollouts_per_pass = 8\n'))
    sixteen = load_config(_write_config(tmp_path / 'sixteen.toml', extra='\nimagination_rollouts_per_pass = 16\n'))
    assert legacy.semantic_fingerprint == explicit.semantic_fingerprint
    assert legacy.semantic_fingerprint != sixteen.semantic_fingerprint
    assert resolved_imagination_budget(sixteen)['rollouts_per_pass'] == 16


@pytest.mark.parametrize('value', ['0', '-1', 'true', '1.5'])
def test_rollout_count_requires_positive_integer(tmp_path, value):
    with pytest.raises(ConfigError, match='imagination_rollouts_per_pass'):
        load_config(_write_config(tmp_path / 'bad.toml', extra=f'\nimagination_rollouts_per_pass = {value}\n'))


def test_sixteen_two_noise_rollouts_select_32_sample_compiled_shape(tmp_path, monkeypatch):
    config = load_config(_write_config(tmp_path / 'config.toml'))
    config = replace(config, resources=replace(config.resources, imagination_rollouts_per_pass=16),
                     method=replace(config.method, disagreement_samples=2))
    dynamics = _small_dynamics()
    trainer = SimpleNamespace(config=config, world_model=SimpleNamespace(dynamics=dynamics))
    assert configure_imagination_inference(trainer)['full_batch'] == 32
    calls = []
    def eager(*args, **kwargs):
        calls.append('eager')
        return DynamicsOutput(torch.ones(1))
    def compile_stub(*args, **kwargs):
        def compiled(*args, **kwargs):
            calls.append('compiled')
            return DynamicsOutput(torch.ones(1))
        return compiled
    monkeypatch.setattr(torch, 'compile', compile_stub)
    wrapper = _VelocityInference(SimpleNamespace(n_edges=3, forward=eager),
        ImaginationInferenceProfile(16, 32768, 16384, 'fp32', True, full_batch=32))
    with torch.no_grad():
        for batch in (32, 16):
            latent = SimpleNamespace(shape=(batch, 1, 3, 64), device=SimpleNamespace(type='cuda'), dtype=torch.float32)
            wrapper(None, packed_latents=latent, flow_time=torch.zeros(batch, 1), condition=object())
    assert calls == ['compiled', 'eager']


def test_sixteen_rollouts_keep_all_seeds_and_noise_independent_of_worker_schedule(monkeypatch):
    monkeypatch.setattr('urban_dreamer.learning.training.IMAGINED_HORIZON', 2)
    trainers = [trainer_case(deadline=60., rollouts=16) for _ in range(2)]
    for trainer in trainers:
        trainer.config.method.disagreement_samples = 2
    try:
        results = [collect_result(trainer, workers) for trainer, workers in zip(trainers, (1, 4))]
        assert results[0][2] == results[1][2]
        for key, value in results[0][0].items():
            torch.testing.assert_close(value, results[1][0][key], rtol=0, atol=0)
        for left, right in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for trainer in trainers:
            assert trainer._last_parallel_work['rollouts'] == 16
            assert trainer._last_parallel_work['windows'] == 32
            assert len(trainer.world_model.calls) == 2
            for batch, noise, events in trainer.world_model.calls:
                assert batch == 16 and noise.shape == (2, 16, 1, 64)
                assert torch.unique(noise.permute(1, 0, 2, 3).reshape(16, -1), dim=0).shape[0] == 16
        for left, right in zip(trainers[0].world_model.calls, trainers[1].world_model.calls):
            torch.testing.assert_close(left[1], right[1], rtol=0, atol=0)
    finally:
        for trainer in trainers:
            close_parallel_imagination(trainer)
