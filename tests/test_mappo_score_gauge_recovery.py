"""Recovery changes only explicit numerical protocol, never training budget/RNG."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import prepare_mappo_score_gauge_recovery as recovery


def payload(version=2):
    return dict(method='mappo', episode=532, best=39.2, format='old', args=dict(method='mappo',
        config='/original/configs/coulommiers.toml', episodes=2500, seed=42, evaluation_argmax=True,
        global_reward_only=True, resume=False, device='cuda:0', collection_device=None),
        method_state=dict(learning_protocol=dict(version=version, reward_scope='global_team_episode_tail'),
                          actor={'weight': torch.arange(12).reshape(3, 4)}, critic={'weight': torch.zeros(4)},
                          actor_optimizer={'state': {0: {'step': torch.tensor(200.), 'exp_avg': torch.ones(12)}}}),
        torch_rng=torch.Generator().manual_seed(1).get_state(), numpy_rng=np.random.RandomState(4).get_state(),
        python_rng=(3, (1, 2, 3), None), cuda_rng=[torch.ones(5056, dtype=torch.uint8)],
        input_hashes={'network': 'old'}, config_fingerprint='unchanged', physics={'traffic_model': 'meso'})


@pytest.mark.parametrize('version', [2, 3])
def test_migration_preserves_every_nonprotocol_field_and_original_input(version):
    old = payload(version)
    baseline = deepcopy(old)
    result = recovery.migrate_checkpoint(old)
    assert recovery.exact_equal(old, baseline)
    assert result['method_state']['learning_protocol'] == dict(version=4,
        reward_scope='global_team_episode_tail', score_gauge=recovery.GAUGE)
    result['method_state']['learning_protocol'] = old['method_state']['learning_protocol']
    assert recovery.exact_equal(result, old)
    result['method_state']['actor']['weight'][0, 0] += 1
    assert not recovery.exact_equal(result, old)


@pytest.mark.parametrize('change', ['method', 'version', 'existing_gauge'])
def test_unsupported_checkpoint_requires_explicit_handling(change):
    old = payload()
    if change == 'method':
        old['method'] = 'dreamer'
    elif change == 'version':
        old['method_state']['learning_protocol']['version'] = 9
    else:
        old['method_state']['learning_protocol']['score_gauge'] = 'unknown'
    with pytest.raises(ValueError):
        recovery.migrate_checkpoint(old)


def source_files():
    return {recovery.POLICY_FILE: '# unchanged prior and critic\n' + recovery.OLD_SCORE,
            recovery.PPO_FILE: '        self.actor = NextHopActor().to(self.device)\n        return protocol\n',
            recovery.TRAINER_FILE: "    parser.add_argument('--ppo-minibatch-size', type=int, default=512)\n"
                                   '    args = parser.parse_args(argv)\n'}


def test_overlay_is_limited_to_reviewed_hunks_and_preserves_old_semantics():
    original = source_files()
    baseline = deepcopy(original)
    result = recovery.overlay_source(original)
    assert original == baseline
    assert result.keys() == original.keys()
    assert result[recovery.POLICY_FILE].startswith('# unchanged prior and critic\n')
    assert 'bias=None' in result[recovery.POLICY_FILE]
    assert recovery.PROTOCOL_SETUP in result[recovery.PPO_FILE]
    assert '--score-gauge' in result[recovery.TRAINER_FILE]
    original[recovery.POLICY_FILE] += recovery.OLD_SCORE
    with pytest.raises(ValueError, match='ambiguous'):
        recovery.overlay_source(original)


def test_resume_cli_keeps_budget_and_all_behavior_parameters():
    saved = payload()['args']
    saved.update(actor_lr=.0002, critic_lr=.0003, evaluation_temperature=.2,
                 save_episodes=True, skip_reference_baselines=True, forbid_revisit=False)
    command = recovery.continuation_argv(saved, Path('/new/source'), Path('/new/run'))
    assert command[command.index('--episodes') + 1] == '2500'
    assert command[command.index('--seed') + 1] == '42'
    assert command[command.index('--config') + 1] == '/new/source/configs/coulommiers.toml'
    assert command[command.index('--score-gauge') + 1] == recovery.GAUGE
    assert '--resume' in command and '--save-episodes' in command and '--evaluation-argmax' in command
    assert '--forbid-revisit' not in command and '--collection-device' not in command
    assert saved['resume'] is False and saved['config'] == '/original/configs/coulommiers.toml'


def test_prepare_has_independent_outputs_exact_checkpoint_and_failed_archive_ancestry(tmp_path):
    source, old_output, destination = (tmp_path / name for name in ('old_source', 'old_output', 'new'))
    for name, text in source_files().items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (source / 'configs').mkdir()
    config = source / 'configs/coulommiers.toml'
    config.write_bytes(b'# original exact bytes\n[inputs]\n')
    (source / 'unrelated.py').write_text('# retain unrelated source\n')
    old_output.mkdir()
    state = payload()
    state['args']['config'] = str(config)
    torch.save(state, old_output / 'latest.pt')
    torch.save(state, old_output / 'best.pt')
    (old_output / 'metrics.jsonl').write_text('{"episode":532}\n')
    archive = old_output / 'episodes/train/episode_000533.npz'
    archive.parent.mkdir(parents=True)
    summary = dict(metadata=dict(episode=533, demand_sha256='unchanged-demand'))
    np.savez(archive, summary_json=np.frombuffer(json.dumps(summary).encode(), dtype=np.uint8))
    before = {p: recovery.digest(p) for p in old_output.rglob('*') if p.is_file()}
    report = recovery.prepare(source, old_output, destination)
    assert report['preservation_verified'] and report['restored_episode'] == 532
    assert report['planned_episode_budget'] == 2500
    assert all(recovery.digest(p) == digest for p, digest in before.items())
    assert (destination / 'source/unrelated.py').read_bytes() == (source / 'unrelated.py').read_bytes()
    assert (destination / 'source/configs/coulommiers.toml').read_bytes() == config.read_bytes()
    restored = torch.load(destination / 'snapshot/latest.pt', weights_only=False)
    restored['method_state']['learning_protocol'] = state['method_state']['learning_protocol']
    assert recovery.exact_equal(restored, state)
    assert not (destination / 'run/metrics.jsonl').is_symlink()
    assert (destination / 'run/metrics.jsonl').read_bytes() == (old_output / 'metrics.jsonl').read_bytes()
    plan = json.loads((destination / 'launch.json').read_text())
    assert plan['stop_after_episode'] == 533 and plan['expected_demand_sha256'] == 'unchanged-demand'
    assert '--episodes' in plan['argv'] and plan['argv'][plan['argv'].index('--episodes') + 1] == '2500'
    compile((destination / 'verify_one_episode.py').read_text(), '<bounded_wrapper>', 'exec')
    with pytest.raises(ValueError, match='overwrite'):
        recovery.prepare(source, old_output, destination)
