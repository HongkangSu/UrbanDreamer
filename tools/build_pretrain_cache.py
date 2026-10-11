"""Build the frozen full-corpus compact cache without touching source archives."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from urban_dreamer.learning.offline_data import OfflineEpisodeDataset, _identity, archive_content_hash
from urban_dreamer.learning.pretrain_cache import CACHE_SCHEMA_VERSION, compact_arrays, file_stamp, member_signature, read_hash_source
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config


_VALIDATOR = None


def atomic_json(path, value):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def initialize_worker(config_path):
    global _VALIDATOR
    torch.set_num_threads(1)
    config = load_config(config_path)
    prepared = validate_prepared_inputs(config)
    _VALIDATOR = object.__new__(OfflineEpisodeDataset)
    _VALIDATOR.config, _VALIDATOR.prepared = config, prepared
    _VALIDATOR.topology = load_traffic_topology(prepared)
    _VALIDATOR._explicit = True


def build_one(row, output_dir, canonical, expected_trajectory):
    source = Path(row['path'])
    data, identity = read_hash_source(source)
    if identity['census_trajectory_signature'] != row['crc_trajectory_candidate']:
        raise ValueError(f'frozen census trajectory content changed: {source}')
    if expected_trajectory is not None and identity['trajectory_hash'] != expected_trajectory:
        raise ValueError(f'frozen census SHA256 trajectory changed: {source}')
    meta = data['metadata']
    demand_hash = next((meta[key] for key in ('demand_sha256', 'demand_hash', 'demand_realization_sha256',
                                              'demand_realization_hash') if meta.get(key)), None)
    if demand_hash != row['demand_hash'] or meta['seed'] != row['seed'] or meta['episode'] != row['episode']:
        raise ValueError(f'frozen census demand/episode metadata changed: {source}')
    _VALIDATOR._validate(source, data)
    result = dict(path=str(source), **identity, demand_group=row['demand_group'],
                  episode=meta['episode'], method=row['method'], root=row['root'],
                  population=meta['total_vehicle_count'], completed=bool(data['completed']),
                  windows=len(data['action']), events=len(data['decisions']), split=row['split'],
                  canonical=canonical)
    if canonical:
        arrays = compact_arrays(data)
        destination = Path(output_dir) / 'episodes' / f"{identity['trajectory_hash']}.npz"
        temporary = destination.with_name(f'.{destination.stem}.{os.getpid()}.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        result.update(cache_path=str(destination.resolve()), cache_content_hash=archive_content_hash(destination),
                      cache_bytes=destination.stat().st_size, cache_source_deadline_seconds=data['deadline_seconds'])
    return result


def make_plan(census_dir, pilot_dir, config_path):
    census = json.loads((census_dir / 'census.json').read_text())
    rows = json.loads((census_dir / 'file_rows.json').read_text())['rows']
    paths = json.loads((census_dir / 'archive_paths.json').read_text())
    if len(rows) != len(paths) or {r['path'] for r in rows} != set(paths):
        raise ValueError('frozen census file lists disagree')
    holdout = set(json.loads((pilot_dir / 'split.json').read_text())['holdout_groups'])
    canonical, expected = {}, {}
    for group in census['verified_duplicate_groups']:
        for path in group['files']:
            canonical[path] = group['files'][0]
            expected[path] = group['trajectory_hash']
    prepared_rows = []
    for row in rows:
        if not row['metadata_compatible'] or not row['demand_hash']:
            raise ValueError('cache source lacks census compatibility or realized demand identity')
        prepared_rows.append({**row, 'split': 'validation' if row['demand_group'] in holdout else 'train'})
    config = load_config(config_path)
    prepared = validate_prepared_inputs(config)
    physical = {key: config.resolved_snapshot()['simulation'][key] for key in (
        'vehicle_max_speed_mps', 'incident_edge', 'incident_start_seconds', 'incident_end_seconds',
        'incident_speed_mps', 'sumo_extra_args', 'demand_duration_seconds')}
    plan = dict(schema_version=CACHE_SCHEMA_VERSION, census_frozen_at=census['frozen_at'],
        config_path=str(config_path.resolve()), contract=dict(city=config.inputs.network_name,
            input_hashes=prepared.input_hashes, physical=physical), sources=prepared_rows,
        holdout_groups=sorted(holdout), canonical=canonical, expected_trajectory=expected,
        expected_unique_trajectories=census['unique_trajectories'])
    from urban_dreamer.traffic.simulation_settings import physics_identity
    physics = physics_identity(config.simulation)
    if physics['traffic_model'] == 'meso':
        plan['contract']['physics_identity'] = physics
    plan['identity'] = _identity(plan)
    return plan


def build_cache(plan, output_dir, *, workers=4):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'episodes').mkdir(exist_ok=True)
    (output_dir / 'receipts').mkdir(exist_ok=True)
    plan_path = output_dir / 'cache_plan.json'
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if old['identity'] != plan['identity']:
            raise ValueError('cache rebuild frozen plan mismatch')
    else:
        atomic_json(plan_path, plan)
    progress_path = output_dir / 'cache_progress.json'
    manifest_path = output_dir / 'cache_manifest.json'
    if manifest_path.exists():
        completed = json.loads(manifest_path.read_text())
        if (completed.get('status') != 'complete' or completed.get('plan_identity') != plan['identity']
                or completed.get('identity') != _identity({k:v for k,v in completed.items() if k != 'identity'})):
            raise ValueError('completed cache manifest does not match the frozen plan')
        for source in completed['sources']:
            path = Path(source['path'])
            if file_stamp(path) != source['source_stamp'] and archive_content_hash(path) != source['content_hash']:
                raise ValueError(f'cache source changed: {path}')
        for entry in completed['entries']:
            if archive_content_hash(Path(entry['cache_path'])) != entry['cache_content_hash']:
                raise ValueError(f"compact episode changed: {entry['cache_path']}")
        print(json.dumps(dict(status='already_complete',identity=completed['identity'],
                              train_episodes=completed['train_episodes'],validation_episodes=completed['validation_episodes'])),flush=True)
        return completed
    results, todo = [], []
    started = time.monotonic()
    for row in plan['sources']:
        receipt = output_dir / 'receipts' / f"{_identity({'path': row['path']})}.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved.pop('plan_identity') != plan['identity']:
                raise ValueError('cache receipt belongs to a different frozen plan')
            path = Path(saved['path'])
            if file_stamp(path) != saved['source_stamp'] and archive_content_hash(path) != saved['content_hash']:
                raise ValueError(f'cache source changed since receipt: {path}')
            if saved.get('cache_path') and archive_content_hash(Path(saved['cache_path'])) != saved['cache_content_hash']:
                raise ValueError('cache episode changed since receipt')
            results.append(saved)
        else:
            todo.append(row)
    atomic_json(progress_path, dict(status='building',completed=len(results),total=len(plan['sources']),workers=workers))
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=initialize_worker, initargs=(plan['config_path'],)) as executor:
        iterator, pending = iter(todo), {}
        def submit():
            row = next(iterator, None)
            if row is None:
                return False
            path = row['path']
            future = executor.submit(build_one, row, str(output_dir), plan['canonical'].get(path, path) == path,
                                     plan['expected_trajectory'].get(path))
            pending[future] = row
            return True
        for _ in range(workers):
            submit()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                row = pending.pop(future)
                result = future.result()
                receipt = output_dir / 'receipts' / f"{_identity({'path': row['path']})}.json"
                atomic_json(receipt, {**result, 'plan_identity': plan['identity']})
                results.append(result)
                submit()
            progress = dict(status='building',completed=len(results),total=len(plan['sources']),workers=workers,
                            elapsed_seconds=time.monotonic()-started)
            atomic_json(progress_path, progress)
            if len(results) % 25 == 0:
                print(json.dumps(progress), flush=True)
    results.sort(key=lambda r:r['path'])
    grouped = {}
    for result in results:
        grouped.setdefault(result['trajectory_hash'], []).append(result)
    entries = []
    for digest, sources in grouped.items():
        if len({source['split'] for source in sources}) != 1:
            raise ValueError('same trajectory spans train and held-out demand groups')
        canonical = [source for source in sources if source.get('cache_path')]
        if not canonical:
            raise ValueError('deduplicated trajectory lacks a compact cache file')
        entries.append(canonical[0])
    entries.sort(key=lambda e:e['path'])
    if len(entries) != plan['expected_unique_trajectories']:
        raise ValueError('strict cache dedup count differs from frozen census')
    manifest = dict(schema_version=CACHE_SCHEMA_VERSION,status='complete',contract=plan['contract'],
        plan_identity=plan['identity'],census_frozen_at=plan['census_frozen_at'],holdout_groups=plan['holdout_groups'],
        sources=results,entries=entries,duplicate_files=len(results)-len(entries),
        train_episodes=sum(e['split']=='train' for e in entries),
        validation_episodes=sum(e['split']=='validation' for e in entries),
        cache_bytes=sum(e['cache_bytes'] for e in entries),elapsed_seconds=time.monotonic()-started)
    manifest['identity'] = _identity(manifest)
    atomic_json(manifest_path,manifest)
    atomic_json(progress_path,{**{k:v for k,v in manifest.items() if k not in {'sources','entries','holdout_groups'}},'completed':len(results),'total':len(results)})
    print(json.dumps({k:manifest[k] for k in ('status','train_episodes','validation_episodes','duplicate_files','cache_bytes','elapsed_seconds')}),flush=True)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--census-dir',type=Path,default=ROOT/'outputs/ingolstadt_pretrain_census_20261002_1148')
    parser.add_argument('--pilot-dir',type=Path,default=ROOT/'outputs/ingolstadt_pretrain_pilot_20261002')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/ingolstadt_custom.toml')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/ingolstadt_pretrain_cache_20261002')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(argv)
    if not 1<=args.workers<=min(32, os.cpu_count() or 1):
        parser.error('workers must be positive and no larger than min(32, available CPUs)')
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';torch.set_num_threads(1)
    plan=make_plan(args.census_dir,args.pilot_dir,args.config)
    build_cache(plan,args.output_dir,workers=args.workers)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
