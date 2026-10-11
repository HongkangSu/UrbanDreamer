"""Physical-cohort MAMBA collects raw events without stale latent transfers."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from test_routing_mamba_baseline import make_method


@pytest.mark.parametrize('explore', [False, True])
def test_no_seed_latent_transfer_and_all_events_retained(monkeypatch, explore):
    method = make_method()
    vehicles = [dict(vehicle_id=f'v{i}', departed=True) for i in range(79)]
    method.begin_episode(SimpleNamespace(vehicles=[SimpleNamespace(**v) for v in vehicles]), seed=9)
    obs = SimpleNamespace(features=np.zeros((79, method.width, 10), dtype=np.float32),
        mask=np.ones((79, method.width), dtype=bool))
    sizes = []
    original = torch.Tensor.cpu
    def record(tensor, *args, **kwargs):
        sizes.append(tensor.numel())
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', record)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        indices, metadata = method.select(vehicles, obs, None, 3., explore=explore)
    finally:
        torch.set_num_threads(previous)
    assert sizes == [32 * 3, 32 * 3, 15 * 3]
    assert len(indices) == len(metadata) == len(method._events[0]['slots']) == 79
    assert set(method._events[0]) == {'time', 'slots', 'obs', 'action', 'mask'}
    assert not hasattr(method, '_seeds')
