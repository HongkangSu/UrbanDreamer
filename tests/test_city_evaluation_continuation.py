from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from tools import continue_city_evaluation_temperature as continuation


@pytest.mark.parametrize('temperature', [.2, 2., .75])
def test_command_changes_only_source_and_evaluation_options(tmp_path, temperature):
    old = tmp_path / 'old'
    entry = dict(id='city_ippo', command=['python', '-u', str(old / 'tools/train.py'),
        '--config', str(old / 'configs/city.toml'), '--output-dir', '/data/unchanged',
        '--episodes', '5000', '--actor-lr', '.0002', '--evaluation-argmax',
        '--evaluation-temperature', '.2'], output_dir='/data/unchanged')
    original = deepcopy(entry)
    actual = continuation.continuation_entry(entry, tmp_path / 'source', tmp_path / 'run', temperature)
    assert entry == original
    assert actual['command'] == ['python', '-u', str(tmp_path / 'source/tools/train.py'),
        '--config', str(tmp_path / 'source/configs/city.toml'), '--output-dir', '/data/unchanged',
        '--episodes', '5000', '--actor-lr', '.0002', '--evaluation-sampled',
        '--evaluation-temperature', str(temperature)]
    assert actual['evaluation_temperature'] == temperature
    assert actual['evaluation_policy'] == f'sampled_temperature_{temperature:g}'


def test_default_command_explicit_argmax_has_no_active_temperature(tmp_path):
    entry = dict(id='city_online_ac', command=['python', '-u', str(tmp_path / 'old/tools/train.py'),
        '--episodes', '5000', '--evaluation-sampled', '--evaluation-temperature', '.2'])
    actual = continuation.continuation_entry(entry, tmp_path / 'new', tmp_path / 'run')
    assert actual['command'] == ['python', '-u', str(tmp_path / 'new/tools/train.py'),
                                 '--episodes', '5000', '--evaluation-argmax']
    assert actual['evaluation_policy'] == 'deterministic'
    assert actual['evaluation_temperature'] is None


@pytest.mark.parametrize('temperature', [0., -1., True, float('inf'), float('nan')])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError, match='finite and positive'):
        continuation.evaluation_label(temperature)


@pytest.mark.parametrize('status', ['failed', 'completed', 'blocked_gpu_capacity'])
def test_terminal_jobs_stay_terminal(status):
    state = dict(status=status, pid=None, attempt=3, required_gpu_mib=6144,
                 exit_code=1, last_error='existing failure')
    assert continuation.continued_job(state, 412) == state


@pytest.mark.parametrize('status', ['running', 'waiting_gpu', 'waiting_host', 'queued_oom'])
def test_unfinished_jobs_resume_durable_cursor_without_changing_budget(status):
    state = dict(status=status, pid=17, attempt=3, required_gpu_mib=6144, oom_retries=2,
                 process_starttime='100', next_eligible=120.)
    actual = continuation.continued_job(state, 412)
    assert actual['status'] == 'queued_recovery'
    assert actual['continuation_episode'] == 412
    assert actual['attempt'] == 3 and actual['oom_retries'] == 2
    assert actual['required_gpu_mib'] == 6144
    assert actual['pid'] is None
    assert state['pid'] == 17


def test_checkpoint_backup_preserves_weights_optimizer_replay_rng_and_clock(tmp_path):
    checkpoint = tmp_path / 'latest.pt'
    state = dict(actor={'weight': torch.arange(12.)}, optimizer={'step': torch.tensor(8)},
                 replay=[{'reward': -4., 'done': False}], rng=torch.get_rng_state(),
                 episode=412, config_fingerprint='unchanged', best=2.8)
    torch.save(state, checkpoint)
    before = checkpoint.read_bytes()
    proof = continuation.archive_checkpoint(checkpoint, tmp_path / 'backup/latest.pt')
    assert checkpoint.read_bytes() == before
    assert Path(proof['backup']).read_bytes() == before
    assert proof['checkpoint_rewritten'] is False
    # Actual training saves atomically, so future updates preserve the old inode.
    temporary = tmp_path / 'latest.tmp'
    torch.save(dict(state, episode=413), temporary)
    temporary.replace(checkpoint)
    assert Path(proof['backup']).read_bytes() == before


def test_checkpoint_backup_rejects_existing_mismatch(tmp_path):
    checkpoint, backup = tmp_path / 'latest.pt', tmp_path / 'backup.pt'
    checkpoint.write_bytes(b'new durable checkpoint')
    backup.write_bytes(b'different previous handoff')
    with pytest.raises(ValueError, match='differs'):
        continuation.archive_checkpoint(checkpoint, backup)


def test_policy_overlay_retains_original_critic_and_temperature_one_training(tmp_path):
    path = tmp_path / 'policy.py'
    text = '''import math
import torch
from torch.distributions import Categorical
class ActionSelection:
    def __init__(self, candidate_index, log_probability, entropy):
        self.candidate_index, self.log_probability, self.entropy = candidate_index, log_probability, entropy
class NextHopActor:
    def forward_logits(self, features, mask):
        return features.masked_fill(~mask, -torch.inf)
    def sample_action(self, features, mask, *, deterministic=False):
        logits = self.forward_logits(features, mask)
        distribution = Categorical(logits=logits)
        selected = logits.argmax(-1) if deterministic else distribution.sample()
        return ActionSelection(selected, distribution.log_prob(selected), distribution.entropy())
class CentralizedLatentValueCritic:
    old_identity = "legacy critic unchanged"
'''
    path.write_text(text)
    original = {}
    exec(text, original)
    continuation.install_actor_temperature(path)
    modified = {}
    exec(path.read_text(), modified)
    assert path.read_text().split('class CentralizedLatentValueCritic:')[1] == text.split('class CentralizedLatentValueCritic:')[1]
    assert 'critic_budget' not in path.read_text()
    features, mask = torch.tensor([[.5, 2., -1., -9.]]), torch.tensor([[True, True, True, False]])
    torch.manual_seed(19)
    before = original['NextHopActor']().sample_action(features, mask)
    torch.manual_seed(19)
    after = modified['NextHopActor']().sample_action(features, mask)
    assert torch.equal(before.candidate_index, after.candidate_index)
    assert torch.equal(before.log_probability, after.log_probability)
    hot = modified['NextHopActor']().sample_action(features, mask, temperature=2.)
    assert hot.entropy > after.entropy
    cold = modified['NextHopActor']().sample_action(features, mask, temperature=.2)
    assert cold.entropy < after.entropy


def test_protocol_keeps_training_config_and_historical_metrics(tmp_path):
    protocol = dict(config={'method': {'actor_entropy_mode': 'annealed'}},
                    evaluation_policy='argmax', args={'actor_lr': .0002},
                    shortest_path_prior={'best_selection': 'deterministic'})
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol))
    metrics = tmp_path / 'metrics.jsonl'
    metrics.write_text('{"policy": "deterministic", "episode": 400}\n')
    before = metrics.read_bytes()
    continuation.update_protocol(tmp_path, 412, tmp_path / 'archive', [10042, 10043, 10044])
    actual = json.loads((tmp_path / 'protocol.json').read_text())
    assert actual['config'] == protocol['config']
    assert actual['args']['actor_lr'] == .0002
    assert actual['evaluation_temperature'] is None
    assert actual['evaluation_policy'] == 'argmax'
    assert actual['shortest_path_prior']['best_selection'] == 'deterministic'
    assert metrics.read_bytes() == before


def test_previous_sampling_best_archived_with_actual_temperature_label(tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    backup = tmp_path / 'best.pt'
    backup.write_bytes(b'T2 evaluation best')
    path = continuation.archive_previous_best(output, backup, 'sampled_temperature_2')
    assert path.name == 'best_sampled_temperature_2.pt'
    assert path.read_bytes() == backup.read_bytes()
    assert not (output / 'best_deterministic.pt').exists()


def test_unreplaced_argmax_best_is_not_relabelled_as_temperature_two(tmp_path):
    import os
    output = tmp_path / 'output'
    output.mkdir()
    backup = tmp_path / 'best.pt'
    backup.write_bytes(b'old argmax best before first T2 evaluation')
    historical = output / 'best_deterministic.pt'
    os.link(backup, historical)
    path = continuation.archive_previous_best(output, backup, 'sampled_temperature_2')
    assert path == historical
    assert not (output / 'best_sampled_temperature_2.pt').exists()
