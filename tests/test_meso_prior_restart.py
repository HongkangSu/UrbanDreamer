"""Protocol replacement must preserve old data and launch a consistent fresh batch."""
import argparse
import json
import os

from tools import restart_baselines_meso as restart
from tools import stop_city_dataset as stop


def test_process_inventory_does_not_count_preparation_command_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(restart, 'command', lambda pid: [str(tmp_path).encode()] if pid == os.getpid() else [])
    assert restart.owned_processes(tmp_path) == []


def test_prepare_replaces_history_mask_with_prior_and_preserves_meso(tmp_path, monkeypatch):
    previous, root = tmp_path / 'previous', tmp_path / 'next'
    previous.mkdir()
    (previous / 'STOPPED_BY_USER').write_text('stopped\n')
    (previous / 'preservation_and_stop.json').write_text(json.dumps(dict(
        inventory_errors=[], remaining_owned_processes=[], total_archived=0)))
    runs = [dict(id=f'{city}_{method}', output_dir=str(previous / city / method))
            for city in restart.CITIES for method in restart.METHODS]
    (previous / 'manifest.json').write_text(json.dumps(dict(runs=runs,
        traffic_model='meso', forbid_revisit=True,
        revisit_policy=dict(enabled=True, node='directed_external_road_edge'),
        policy_prior=dict(enabled=False, initial_weight=0., decay='cosine',
                          random_control='uniform legal actions; no prior'))))
    monkeypatch.setattr(restart, 'owned_processes', lambda _: [])
    def snapshot(destination):
        source = destination / 'source'
        source.mkdir()
        return source
    monkeypatch.setattr(restart, 'snapshot', snapshot)
    args = argparse.Namespace(previous_root=previous, output_root=root,
        device=[f'{r["id"]}=cpu' for r in runs], forbid_revisit=False,
        sp_prior_weight=1., sp_prior_temperature=30.,
        sp_prior_hold_episodes=200, sp_prior_off_episode=1000)
    restart.prepare(args)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert len(manifest['runs']) == 15
    assert not manifest['forbid_revisit'] and not manifest['revisit_policy']['enabled']
    assert manifest['policy_prior']['enabled']
    assert manifest['policy_prior']['best_selection'] == 'zero_prior'
    assert manifest['migration']['physics_changed_from'] == 'meso'
    assert manifest['migration']['previous_forbid_revisit'] is True
    for run in manifest['runs']:
        command = run['command']
        assert '--resume' not in command and '--forbid-revisit' not in command
        assert run['simulation_settings']['traffic_model'] == 'meso'
        if run['method'] == 'random':
            assert '--sp-prior-weight' not in command
        else:
            assert command[command.index('--sp-prior-weight') + 1] == '1.0'
            assert command[command.index('--sp-prior-off-episode') + 1] == '1000'
    old_manifest = json.loads((previous / 'manifest.json').read_text())
    assert old_manifest['forbid_revisit'] is True
    assert old_manifest['superseded_by']['root'] == str(root)


def test_stop_preserves_run_that_has_not_saved_first_archive(tmp_path, monkeypatch):
    state = tmp_path / 'random_state.json'
    state.write_text(json.dumps(dict(status='starting')))
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(runs=[dict(
        id='city_random', city='city', method='random', output_dir=str(tmp_path / 'random'),
        state=str(state), tmux_session='unused')])))
    monkeypatch.setattr(stop.time, 'sleep', lambda _: None)
    monkeypatch.setattr(stop.subprocess, 'run', lambda *a, **kw: None)
    monkeypatch.setattr(stop.sys, 'argv', ['stop_city_dataset.py', '--root', str(tmp_path)])
    stop.main()
    saved = json.loads((tmp_path / 'preservation_and_stop.json').read_text())
    assert saved['total_archived'] == 0
    assert saved['inventory_errors'] == saved['remaining_owned_processes'] == []
    assert (tmp_path / 'STOPPED_BY_USER').is_file()
