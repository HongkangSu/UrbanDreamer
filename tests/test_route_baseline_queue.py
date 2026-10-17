"""Fresh route queues preserve physical inputs, budgets and reference fairness."""
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from tools import queued_city_baselines as queue
from tools import monitor_route_online_ac as monitor
from tools.monitor_route_online_ac import paired_summary
from urban_dreamer.workflow.config import load_config


def test_route_queue_has_27_fresh_training_jobs_prioritizes_ac_and_matches_references(tmp_path, monkeypatch):
    def snapshot(root):
        source = root/'source'
        (source/'configs').mkdir(parents=True)
        hashes = {}
        for city in queue.CITIES:
            path = source/'configs'/f'{city}.toml'
            shutil.copyfile(queue.ROOT/'configs'/f'{city}.toml',path)
            hashes[f'configs/{city}.toml'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (source/'networks').symlink_to(queue.ROOT/'networks',target_is_directory=True)
        queue.write_json(root/'source_hashes.json',hashes)
        return source
    monkeypatch.setattr('tools.run_city_baseline_dataset.snapshot',snapshot)
    root = tmp_path/'batch'
    queue.prepare(SimpleNamespace(output_root=root, episodes=1000,poll_seconds=10.,
        gpu_headroom_mib=1024,min_free_gb=.001, routing_mode='current_to_dest',
        methods=(*queue.LEARNED,'dta')))
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['total_training_episodes']==27000
    assert len(manifest['runs'])==33
    assert [entry['method'] for entry in manifest['runs'][:3]]==['online_ac']*3
    assert [entry['city'] for entry in manifest['runs'][:3]]==list(queue.CITIES)
    hashes=json.loads((root/'source_hashes.json').read_text())
    for city in queue.CITIES:
        path=root/'source/configs'/f'{city}.toml'
        config=load_config(path)
        assert config.simulation.routing_mode=='current_to_dest'
        assert config.simulation.traffic_model=='meso'
        assert config.simulation.backend=='libsumo'
        assert config.run.online_training_episodes==1000
        assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[f'configs/{city}.toml']
        assert path.resolve()!=queue.ROOT/'configs'/f'{city}.toml'
    for entry in manifest['runs']:
        command=entry['command']
        assert Path(entry['output_dir']).is_relative_to(root)
        assert '--resume' not in command
        if entry['method'].endswith('_ref'):
            assert command[command.index('--output-dir')+1]==entry['output_dir']
            assert command[command.index('--policy')+1]==entry['method'][:-4]
        else:
            assert entry['episode_budget']==1000
            assert command[command.index('--episodes')+1]=='1000'
            assert '--evaluation-argmax' in command
            assert command[command.index('--eval-every')+1]==('20' if entry['method']=='online_ac' else '100')
            if entry['method'] in {'mappo','ippo'}:
                assert command[command.index('--score-gauge')+1]=='pre_head_legal_reference_v1'


def rows(mean=200.,unfinished=0):
    return [dict(seed=10042+i,demand_sha256=f'demand{i}',vehicle_count=100,
        started_vehicle_count=100,cumulative_vehicle_seconds=mean*100,
        unfinished=unfinished,p95_travel_seconds=300.) for i in range(3)]


def test_live_comparison_claims_improvement_only_for_identical_complete_demands():
    result=paired_summary(rows(180.),rows(200.))
    assert result['online_ac_beats_dynamic_shortest'] is True
    assert result['mean_reduction_percent']==pytest.approx(10.)
    ac=rows(180.)
    for row in ac:
        row['demand_realization_sha256']=row.pop('demand_sha256')
    assert paired_summary(ac,rows())['online_ac_beats_dynamic_shortest'] is True
    assert paired_summary(rows(180.,unfinished=1),rows(200.))['online_ac_beats_dynamic_shortest'] is None
    assert paired_summary(rows()[:2],rows()) is None
    tampered=rows()
    tampered[1]['demand_sha256']='another-demand'
    with pytest.raises(ValueError,match='demands differ'):
        paired_summary(tampered,rows())


def test_aggressive_admission_reduces_pending_reservations_without_hiding_actual_use():
    gpu=dict(index=0,free_mib=10000)
    running=[dict(gpu_index=0,required_gpu_mib=8000,pid=123)]
    assert queue.effective_gpu_free_mib(gpu,running,{123:2000},512)==3488
    assert queue.effective_gpu_free_mib(gpu,running,{123:2000},512,.5)==6488
    assert queue.choose_gpu(8000,[gpu],running,{123:2000},512) is None
    assert queue.choose_gpu(8000,[gpu],running,{123:2000},512,.5)==0
    assert queue.choose_gpu(1000,[dict(index=0,free_mib=500)],[],{},512,.5) is None


def test_host_admission_counts_actual_ram_and_reserves_only_future_growth():
    # The OS available value already excludes the resident 2,000 MiB.
    assert queue.effective_host_free_mib(10000, [(8000, 2000)], 1000) == 3000
    assert queue.effective_host_free_mib(10000, [(8000, 2000)], 1000, .5) == 6000
    assert queue.effective_host_free_mib(10000, [(8000, 9000)], 1000, .5) == 9000
    assert queue.effective_host_free_mib(500, [], 1000, .5) == 0


@pytest.mark.parametrize('completed_seeds', [0, 2, 3])
def test_monitor_reports_evaluation_progress_until_checkpoint_commit(tmp_path, monkeypatch, completed_seeds):
    run = dict(city='ingo', method='online_ac', command=['runner', '--eval-every', '20'])
    queue.write_json(tmp_path/'manifest.json', dict(routing_mode='current_to_dest',
        episode_budget=1000, runs=[run]))
    (tmp_path/'ingo/dynamic_shortest_ref').mkdir(parents=True)
    queue.write_json(tmp_path/'ingo/dynamic_shortest_ref/rows.json', rows())
    previous = [dict(row, phase='evaluation', policy='deterministic', episode=20) for row in rows(180.)]
    partial = [dict(row, phase='evaluation', policy='deterministic', episode=40)
               for row in rows(220.)[:completed_seeds]]
    records = previous + [dict(phase='train', episode=40)] + partial
    if partial:
        records.append(partial[0])  # A retried seed must not inflate progress.
    path = tmp_path/'ingo/online_ac/metrics.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in records))
    monkeypatch.setattr(monitor, 'durable_episode', lambda _: 39)

    report = monitor.build_report(tmp_path)['cities']['ingo']
    assert report['committed_training_episode'] == 39
    assert report['logged_training_episode'] == 40
    assert report['evaluation_progress'] == dict(episode=40,
        completed_seeds=[10042+i for i in range(completed_seeds)],
        completed_seed_count=completed_seeds, expected_seed_count=3)
    assert report['latest_evaluation']['episode'] == (40 if completed_seeds == 3 else 20)

    if completed_seeds == 3:
        monkeypatch.setattr(monitor, 'durable_episode', lambda _: 40)
        committed = monitor.build_report(tmp_path)['cities']['ingo']
        assert 'evaluation_progress' not in committed
        assert committed['latest_evaluation']['episode'] == 40
