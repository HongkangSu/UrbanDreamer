import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fifteen_independent_5000_episode_jobs_with_archives_and_devices(tmp_path):
    tool = load_tool('run_city_baseline_dataset')
    args = argparse.Namespace(cities=list(tool.CITIES), methods=list(tool.METHODS),
        devices=['cuda:0', 'cuda:1'], episodes=5000, seed=42, checkpoint_every=25,
        eval_every=100, eval_seeds=3)
    runs = tool.build_runs(tmp_path, tmp_path / 'source', args, 'test')
    assert len(runs) == len({run['id'] for run in runs}) == 15
    assert len({run['output_dir'] for run in runs}) == 15
    assert len({run['tmux_session'] for run in runs}) == 15
    for run in runs:
        command = run['command']
        assert command[command.index('--episodes') + 1] == '5000'
        assert command[command.index('--session') + 1] == 'event'
        assert command[command.index('--backend') + 1] == 'libsumo'
        assert command[command.index('--seed') + 1] == '42'
        if run['method'] == 'random':
            assert 'collect_random_episodes.py' in command[2]
            assert run['device'] == 'cpu'
        else:
            assert '--save-episodes' in command and '--skip-initial-evaluation' in command
    assert {run['device'] for run in runs if run['method'] == 'dreamer'} == {'cuda:0', 'cuda:1'}


def test_status_counts_archives_not_duplicate_metric_lines(tmp_path):
    tool = load_tool('run_city_baseline_dataset')
    output = tmp_path / 'city' / 'random'
    archives = output / 'episodes' / 'train'
    archives.mkdir(parents=True)
    (archives / 'episode_000001.npz').write_bytes(b'one')
    (archives / '.episode_000002.npz.tmp').write_bytes(b'partial')
    (output / 'metrics.jsonl').write_text('{"episode":1}\n' * 3)
    state = tmp_path / 'state.json'
    state.write_text('{"status":"running","pid":1}')
    manifest = dict(episode_budget=5000, total_training_episodes=5000,
        runs=[dict(city='city', method='random', device='cpu', output_dir=str(output), state=str(state))])
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    status = tool.report(argparse.Namespace(output_root=tmp_path))
    assert status['archived_episodes'] == 1
    assert status['runs'][0]['archive_bytes'] == 3
    assert status['runs'][0]['latest_archived'] == 1


def test_prior_protocol_applies_only_to_learned_methods(tmp_path):
    tool = load_tool('run_city_baseline_dataset')
    args = argparse.Namespace(cities=list(tool.CITIES), methods=list(tool.METHODS),
        devices=['cuda:0', 'cuda:1'], episodes=5000, seed=42, checkpoint_every=10,
        eval_every=100, eval_seeds=3, sp_prior_weight=1., sp_prior_temperature=30.,
        sp_prior_hold_episodes=200, sp_prior_off_episode=1000)
    runs = tool.build_runs(tmp_path, tmp_path / 'source', args, 'prior')
    for run in runs:
        command = run['command']
        if run['method'] == 'random':
            assert not any(arg.startswith('--sp-prior') for arg in command)
        else:
            assert command[command.index('--sp-prior-weight') + 1] == '1.0'
            assert command[command.index('--sp-prior-temperature') + 1] == '30.0'
            assert command[command.index('--sp-prior-hold-episodes') + 1] == '200'
            assert command[command.index('--sp-prior-off-episode') + 1] == '1000'
