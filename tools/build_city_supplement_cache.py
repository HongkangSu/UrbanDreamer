"""Reuse verified old compact episodes and build the frozen supplemental cache."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_pretrain_cache import atomic_json, build_cache, make_plan
from urban_dreamer.learning.offline_data import _identity, archive_content_hash
from urban_dreamer.learning.pretrain_cache import file_stamp


def seed_receipts(plan, previous_manifest, output):
    """Reuse only old sources that remain canonical in the new frozen plan."""
    if (previous_manifest.get('status') != 'complete'
            or previous_manifest.get('identity') != _identity({
                key: value for key, value in previous_manifest.items() if key != 'identity'})):
        raise ValueError('previous compact cache manifest is incomplete or changed')
    if previous_manifest['contract'] != plan['contract']:
        raise ValueError('previous and supplemental cache contracts differ')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    receipts = output / 'receipts'
    receipts.mkdir(exist_ok=True)
    plan_path = output / 'cache_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())['identity'] != plan['identity']:
            raise ValueError('supplement cache contains a different frozen plan')
    else:
        atomic_json(plan_path, plan)
    old_sources = {source['path']: source for source in previous_manifest['sources']}
    started = time.monotonic()
    stats = dict(reused_sources=0, reused_compact_bytes=0, existing_receipts=0,
                 old_noncanonical_skipped=0, changed_canonical_skipped=0,
                 new_sources=sum(row['path'] not in old_sources for row in plan['sources']))
    def publish_progress():
        atomic_json(output / 'cache_progress.json', dict(
            status='seeding', completed=stats['reused_sources'], total=len(plan['sources']),
            elapsed_seconds=time.monotonic()-started, **stats))
    publish_progress()
    for row in plan['sources']:
        old = old_sources.get(row['path'])
        if old is None:
            continue
        if not old.get('canonical') or not old.get('cache_path'):
            stats['old_noncanonical_skipped'] += 1
            continue
        if plan['canonical'].get(row['path'], row['path']) != row['path']:
            stats['changed_canonical_skipped'] += 1
            continue
        for key in ('path', 'demand_group', 'episode', 'method', 'root', 'split'):
            if old[key] != row[key]:
                raise ValueError(f"old cache source identity differs: {row['path']} ({key})")
        if (row['demand_hash'] != old['demand_group']
                or row['crc_trajectory_candidate'] != old['census_trajectory_signature']):
            raise ValueError(f"old cache trajectory/demand identity differs: {row['path']}")
        expected = plan['expected_trajectory'].get(row['path'])
        if expected is not None and expected != old['trajectory_hash']:
            raise ValueError(f"old cache trajectory SHA256 differs: {row['path']}")
        source, compact = Path(old['path']), Path(old['cache_path'])
        stamp = file_stamp(source)
        if stamp != old['source_stamp'] and archive_content_hash(source) != old['content_hash']:
            raise ValueError(f'old canonical source content changed: {source}')
        if archive_content_hash(compact) != old['cache_content_hash']:
            raise ValueError(f'old immutable compact episode changed: {compact}')
        saved = {**old, 'source_stamp': stamp, 'plan_identity': plan['identity']}
        destination = receipts / f"{_identity({'path': row['path']})}.json"
        if destination.exists():
            existing = json.loads(destination.read_text())
            if existing != saved:
                raise ValueError(f'existing supplemental receipt differs: {destination}')
            stats['existing_receipts'] += 1
        else:
            atomic_json(destination, saved)
        stats['reused_sources'] += 1
        stats['reused_compact_bytes'] += compact.stat().st_size
        if stats['reused_sources'] % 25 == 0:
            publish_progress()
    stats['sources_to_build'] = len(plan['sources']) - stats['reused_sources']
    atomic_json(output / 'reuse_summary.json', stats)
    publish_progress()
    return stats


def prepare_and_build(city_dir, previous_city_dir, *, workers=2):
    city_dir, previous_city_dir = Path(city_dir).resolve(), Path(previous_city_dir).resolve()
    if city_dir == previous_city_dir or previous_city_dir in city_dir.parents:
        raise ValueError('supplement cache must be isolated from the previous run')
    plan = make_plan(city_dir / 'census', city_dir / 'pilot', city_dir / 'base_config.toml')
    previous = json.loads((previous_city_dir / 'cache/cache_manifest.json').read_text())
    stats = seed_receipts(plan, previous, city_dir / 'cache')
    print(json.dumps(dict(stage='reuse_verified', **stats)), flush=True)
    return build_cache(plan, city_dir / 'cache', workers=workers)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city-dir', required=True, type=Path)
    parser.add_argument('--previous-city-dir', required=True, type=Path)
    parser.add_argument('--workers', default=2, type=int)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= min(32, os.cpu_count() or 1):
        parser.error('workers must be between 1 and min(32, available CPUs)')
    previous, city = args.previous_city_dir.resolve(), args.city_dir.resolve()
    if city == previous or previous in city.parents:
        parser.error('supplement cache must be isolated from the previous run')
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    try:
        prepare_and_build(args.city_dir, args.previous_city_dir, workers=args.workers)
    except Exception as error:
        output = args.city_dir.resolve() / 'cache'
        output.mkdir(parents=True, exist_ok=True)
        path = output / 'cache_progress.json'
        try:
            progress = json.loads(path.read_text()) if path.exists() else {}
        except (OSError, ValueError):
            progress = {}
        atomic_json(path, {**progress, 'status': 'failed',
                          'error': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    main()
