from types import SimpleNamespace

import torch

from test_toy_model_free_actor_critic import args, module, rows, windows


def test_update_timing_preserves_actor_and_critic_update(tmp_path, monkeypatch):
    tool = module()
    runner = tool.RealActorCritic(args(tmp_path))
    reference = tool.RealActorCritic(args(tmp_path))
    events = rows()
    collected = windows(runner, [-.5, -2.])
    expected = reference.update(events, collected)
    ticks = iter(range(20))
    monkeypatch.setattr(tool, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))
    actual = runner.update(events, collected)
    assert actual['actor_update_seconds'] == 1.
    assert actual['critic_update_seconds'] == 2.
    assert actual['learning_seconds'] == 7.
    for key in expected:
        if not key.endswith('_seconds'):
            assert actual[key] == expected[key]
    for name in ('actor', 'global_critic'):
        for key, value in getattr(runner, name).state_dict().items():
            torch.testing.assert_close(value, getattr(reference, name).state_dict()[key], rtol=0, atol=0)
