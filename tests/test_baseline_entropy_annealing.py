"""Actor baselines use the restored episode's common entropy coefficient."""
from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch
from torch.distributions import Categorical

from test_toy_baselines import module
from test_toy_baselines_model_based import make_args
from urban_dreamer.learning.baselines import common
from urban_dreamer.learning.baselines.gnac import GraphActorCriticMethod
from urban_dreamer.learning.baselines.dreamer import DreamerMethod
from urban_dreamer.learning.baselines.mbpo import MbpoMethod
from urban_dreamer.learning.baselines.mamba import MambaMethod, PPO_EPOCHS
from urban_dreamer.learning.baselines.ppo import IppoMethod, MappoMethod
from urban_dreamer.workflow.config import MethodConfig


@pytest.mark.parametrize('method_class', [GraphActorCriticMethod, DreamerMethod,
    MbpoMethod, MambaMethod, IppoMethod, MappoMethod])
def test_annealed_coefficient_uses_full_restored_progress(method_class):
    method = method_class.__new__(method_class)
    config = replace(MethodConfig(), actor_entropy_mode='annealed')
    tool = module()
    first = tool.apply_episode_entropy(method, config,
        completed_episodes=0, total_episodes=5000)
    assert first['entropy_coefficient'] == .002
    resumed = tool.apply_episode_entropy(method, config,
        completed_episodes=2250, total_episodes=5000)
    assert method.entropy_coef == pytest.approx(.00105)
    assert resumed['entropy_schedule_progress'] == .45
    final = tool.apply_episode_entropy(method, config,
        completed_episodes=4000, total_episodes=5000)
    assert final['entropy_coefficient'] == .0001
    assert method.entropy_coef == .0001


@pytest.mark.parametrize('mode, expected', [('population', .002), ('legacy', .002), ('off', 0.)])
def test_baseline_historical_modes_keep_fixed_coefficient(mode, expected):
    method = GraphActorCriticMethod.__new__(GraphActorCriticMethod)
    result = module().apply_episode_entropy(method,
        replace(MethodConfig(), actor_entropy_mode=mode),
        completed_episodes=4900, total_episodes=5000)
    assert method.entropy_coef == expected
    assert result['entropy_coefficient'] == expected


def test_q_learning_has_no_fabricated_entropy_loss():
    method = SimpleNamespace(actor=object())
    result = module().apply_episode_entropy(method,
        replace(MethodConfig(), actor_entropy_mode='annealed'),
        completed_episodes=2250, total_episodes=5000)
    assert result['entropy_mode'] == 'not_applicable'
    assert result['entropy_coefficient'] == 0.
    assert not hasattr(method, 'entropy_coef')


def test_mbpo_all_updates_receive_frozen_episode_coefficient(tmp_path, monkeypatch):
    import urban_dreamer.learning.baselines.mbpo as mbpo
    runner = common.EpisodeRunner(make_args(tmp_path, method='mbpo'))
    method = MbpoMethod(runner.args, runner)
    module().apply_episode_entropy(method,
        replace(MethodConfig(), actor_entropy_mode='annealed'),
        completed_episodes=4000, total_episodes=5000)
    coefficients = []
    monkeypatch.setattr(mbpo, 'policy_gradient_step',
        lambda **kwargs: coefficients.append(kwargs['entropy_coef']) or {'actor_loss': 0.})
    method._policy_updates([{'window': 0}], torch.zeros(1))
    assert coefficients == [.0001] * mbpo.POLICY_UPDATES_PER_EPISODE


def test_mamba_all_ppo_epochs_use_scheduled_coefficient():
    method = MambaMethod.__new__(MambaMethod)
    method.args = SimpleNamespace(global_reward_only=True)
    logits = torch.tensor([[1., -.5]], requires_grad=True)
    distribution = Categorical(logits=logits)
    method.actor = SimpleNamespace(prior_weight=0.,
        routing_distribution=lambda *args, **kwargs: distribution)
    method.actor_optimizer = object()
    method.actor.parameters = lambda: [logits]
    method.entropy_coef = .0001
    losses = []
    method._step = lambda optimizer, loss, parameters: losses.append(float(loss.detach()))
    record = dict(step=0, action=torch.tensor([0]), latent=None, candidates=None,
        mask=None, old_log_prob=distribution.log_prob(torch.tensor([0])).detach())
    method._ppo_update([record], torch.zeros(1))
    expected = -method.entropy_coef * float(distribution.entropy().mean().detach())
    assert losses == pytest.approx([expected] * PPO_EPOCHS)


@pytest.mark.parametrize('cls', [IppoMethod, MappoMethod])
def test_ppo_runtime_schedule_preserves_strict_checkpoint_protocol(cls):
    from test_routing_ppo_baselines import method, data
    model = method(cls)
    model.on_episode(*data(model), episode=1)
    original = model.state_dict()
    config = replace(MethodConfig(), actor_entropy_mode='annealed')
    module().apply_episode_entropy(model, config,
        completed_episodes=4000, total_episodes=5000)
    assert model.learning_protocol() == original['learning_protocol']
    restored = method(cls)
    restored.load_state_dict(model.state_dict())
    module().apply_episode_entropy(restored, config,
        completed_episodes=4000, total_episodes=5000)
    assert restored.entropy_coef == .0001
    torch.testing.assert_close(restored.actor.state_dict(), model.actor.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(restored.actor_optimizer.state_dict(),
        model.actor_optimizer.state_dict(), rtol=0, atol=0)


def test_gnac_actor_step_receives_scheduled_coefficient(tmp_path, monkeypatch):
    import urban_dreamer.learning.baselines.gnac as gnac
    runner = common.EpisodeRunner(make_args(tmp_path, method='gnac'))
    method = GraphActorCriticMethod(runner.args, runner)
    module().apply_episode_entropy(method,
        replace(MethodConfig(), actor_entropy_mode='annealed'),
        completed_episodes=4000, total_episodes=5000)
    coefficients = []
    monkeypatch.setattr(gnac, 'global_critic_update', lambda **kwargs: (torch.zeros(1), {}))
    monkeypatch.setattr(gnac, 'policy_gradient_step',
        lambda **kwargs: coefficients.append(kwargs['entropy_coef']) or {'actor_loss': 0.})
    method.on_episode([{'window': 0}], [{}], episode=4001)
    assert coefficients == [.0001]


def test_gnac_cli_extends_old_checkpoint_without_restarting_entropy(tmp_path):
    import json
    from micro_config_fixture import write_micro_config
    from test_toy_baselines import sumo_available
    if not sumo_available():
        pytest.skip('SUMO unavailable')
    config = write_micro_config(tmp_path)
    text = config.read_text()
    import re
    text = re.sub(r'^actor_entropy_mode\s*=.*$', 'actor_entropy_mode = "annealed"', text, flags=re.MULTILINE)
    if 'actor_entropy_mode' not in text:
        text = text.replace('[method]', '[method]\nactor_entropy_mode = "annealed"')
    config.write_text(text)
    output = tmp_path / 'extended'
    command = ['--method', 'gnac', '--config', str(config), '--output-dir', str(output),
               '--eval-every', '1', '--eval-seeds', '1', '--vehicle-limit', '8',
               '--skip-reference-baselines', '--skip-initial-evaluation']
    tool = module()
    assert tool.main(command + ['--episodes', '1']) == 0
    first = torch.load(output / 'latest.pt', weights_only=False)
    assert first['args'].get('entropy_schedule_episodes') is None
    with pytest.raises(ValueError, match='entropy schedule'):
        tool.main(command + ['--episodes', '2', '--resume'])
    assert tool.main(command + ['--episodes', '2', '--entropy-schedule-episodes', '1', '--resume']) == 0
    final = torch.load(output / 'latest.pt', weights_only=False)
    assert final['episode'] == 2
    assert final['args']['episodes'] == 2
    assert final['args']['entropy_schedule_episodes'] == 1
    assert final['config_fingerprint'] == first['config_fingerprint']
    training = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()
                if json.loads(line).get('phase') == 'train']
    assert [row['episode'] for row in training] == [1, 2]
    assert training[-1]['entropy_coefficient'] == .0001
    assert training[-1]['entropy_total_episodes'] == 1
