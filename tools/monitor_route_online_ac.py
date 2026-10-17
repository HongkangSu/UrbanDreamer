"""Compare fixed-demand Online AC evaluations with matching dynamic shortest controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.queued_city_baselines import durable_episode, write_json


def read_records(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # The producer may still be writing the final record.
    return result


def paired_summary(records, reference):
    refs = {row['seed']: row for row in reference}
    if len(records) != 3 or any(row['seed'] not in refs for row in records):
        return None
    pairs = [(row, refs[row['seed']]) for row in records]
    if any(row.get('demand_sha256', row.get('demand_realization_sha256')) != ref['demand_sha256']
           or row['vehicle_count'] != ref['vehicle_count']
           for row, ref in pairs):
        raise ValueError('Online AC and dynamic shortest evaluation demands differ')
    def aggregate(rows):
        population = sum(row['vehicle_count'] for row in rows)
        starts = sum(row['started_vehicle_count'] for row in rows)
        seconds = sum(row['cumulative_vehicle_seconds'] for row in rows)
        unfinished = sum(row['unfinished'] for row in rows)
        p95 = [row.get('p95_travel_seconds') for row in rows]
        return dict(vehicle_count=population, unfinished=unfinished,
            completion_rate=1.-unfinished/max(population,1),
            mean_seconds=seconds/max(starts,1),
            mean_episode_p95_seconds=(sum(p95)/len(p95) if all(value is not None for value in p95) else None))
    ac, dynamic = aggregate([row for row,_ in pairs]), aggregate([ref for _,ref in pairs])
    complete = ac['unfinished'] == dynamic['unfinished'] == 0
    return dict(online_ac=ac, dynamic_shortest=dynamic, paired_demands_verified=True,
        mean_reduction_percent=(100*(1-ac['mean_seconds']/dynamic['mean_seconds']) if complete else None),
        online_ac_beats_dynamic_shortest=(ac['mean_seconds'] < dynamic['mean_seconds'] if complete else None),
        note='Full travel-time comparison requires both policies to complete all trips; p95 is the mean of episode p95 values.')


def build_report(root):
    manifest = json.loads((root/'manifest.json').read_text())
    cities = sorted({entry['city'] for entry in manifest['runs'] if entry['method']=='online_ac'})
    report = dict(updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        routing_mode=manifest['routing_mode'], episode_budget=manifest['episode_budget'], cities={})
    for city in cities:
        reference_path = root/city/'dynamic_shortest_ref/rows.json'
        reference = json.loads(reference_path.read_text()) if reference_path.exists() else []
        records = read_records(root/city/'online_ac/metrics.jsonl')
        groups = {}
        for record in records:
            if record.get('phase') in ('evaluation','holdout_final','holdout_best') and record.get('policy')=='deterministic':
                key = (record['phase'], record['episode'])
                # Retried durable episodes must not multiply a fixed-demand evaluation.
                groups.setdefault(key,{})[record['seed']] = record
        checkpoints = []
        for (phase,episode), values in sorted(groups.items()):
            summary = paired_summary(list(values.values()), reference)
            if summary is not None:
                checkpoints.append(dict(phase=phase, episode=episode, **summary))
        training = [record for record in records if record.get('phase')=='train']
        run = next(entry for entry in manifest['runs'] if entry['city']==city and entry['method']=='online_ac')
        entry = dict(committed_training_episode=durable_episode(run),
            logged_training_episode=max((record['episode'] for record in training),default=0),
            reference_demands_completed=len(reference), evaluations=checkpoints)
        command = run.get('command', [])
        if '--eval-every' in command:
            interval = int(command[command.index('--eval-every') + 1])
            episode = entry['logged_training_episode']
            if interval > 0 and episode > entry['committed_training_episode'] and episode % interval == 0:
                seeds = sorted(groups.get(('evaluation', episode), {}))
                entry['evaluation_progress'] = dict(episode=episode,
                    completed_seeds=seeds, completed_seed_count=len(seeds), expected_seed_count=3)
        evaluations = [row for row in checkpoints if row['phase']=='evaluation']
        if evaluations:
            entry['latest_evaluation'] = evaluations[-1]
            complete = [row for row in evaluations if row['online_ac_beats_dynamic_shortest'] is not None]
            if complete:
                entry['best_validation_evaluation'] = min(complete,key=lambda row:row['online_ac']['mean_seconds'])
        holdouts = [row for row in checkpoints if row['phase'].startswith('holdout')]
        if holdouts:
            entry['holdout_evaluations'] = holdouts
        report['cities'][city] = entry
    write_json(root/'online_ac_vs_dynamic_shortest.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',required=True,type=Path)
    parser.add_argument('--poll-seconds',type=float,default=30.)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args(argv)
    if args.poll_seconds<=0:
        parser.error('--poll-seconds must be positive')
    while True:
        report=build_report(args.output_root.resolve())
        print(json.dumps({city:dict(episode=row['committed_training_episode'],
            reference_demands=row['reference_demands_completed'],
            evaluation_progress=row.get('evaluation_progress'),
            latest_comparison=row.get('latest_evaluation')) for city,row in report['cities'].items()}),flush=True)
        if args.once or (args.output_root/'final_status.json').exists():
            return
        time.sleep(args.poll_seconds)


if __name__=='__main__':
    main()
