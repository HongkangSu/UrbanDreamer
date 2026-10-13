"""Freeze additional committed city data without changing the trained split."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.prepare_city_pretrain import (
    COLLECTION, LIVE_MANIFEST, PHYSICAL, export_relative, freeze_committed,
    refreshed, representative_rows, scan_metadata, verify_duplicates, write_json,
)
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.simulation_settings import physics_identity
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config


def anchored_holdout(rows, previous_rows, previous_holdout, *, seed=42):
    """Never reassign an old group; assign unseen groups by an absolute hash cut."""
    old_groups = {row['demand_group'] for row in previous_rows}
    holdout = set(previous_holdout)
    if not holdout <= old_groups:
        raise ValueError('previous holdout contains groups absent from the full census')
    for group in {row['demand_group'] for row in rows} - old_groups:
        digest = hashlib.sha256(f'city-pretrain:{seed}:{group}'.encode()).digest()
        if int.from_bytes(digest, 'big') * 100 < 15 * (1 << 256):
            holdout.add(group)
    return holdout


def previous_partition(previous):
    rows = json.loads((previous / 'census/file_rows.json').read_text())['rows']
    census = json.loads((previous / 'census/census.json').read_text())
    paths = json.loads((previous / 'census/archive_paths.json').read_text())
    if len(rows) != len(paths) or {r['path'] for r in rows} != set(paths):
        raise ValueError('previous full census membership is inconsistent')
    split = json.loads((previous / 'pilot/split.json').read_text())
    holdout = set(split['holdout_groups'])
    anchored_holdout(rows, rows, holdout, seed=split['seed'])
    # Check the full cache, rather than only the representative codec sample.
    plan = json.loads((previous / 'cache/cache_plan.json').read_text())
    by_path = {r['path']: r for r in rows}
    if set(by_path) != {r['path'] for r in plan['sources']}:
        raise ValueError('previous cache and census membership differ')
    for source in plan['sources']:
        row = by_path[source['path']]
        expected = 'validation' if row['demand_group'] in holdout else 'train'
        if source['demand_group'] != row['demand_group'] or source['split'] != expected:
            raise ValueError('previous full-corpus partition disagrees with the anchor')
    return rows, census, holdout, split['seed']


def verify_old_rows(rows, previous_rows):
    """Compaction may change container bytes, but not trajectory or identities."""
    by_path = {r['path']: r for r in rows}
    for old in previous_rows:
        fresh = by_path.get(old['path'])
        if fresh is None:
            raise ValueError('supplement discarded a previous archive')
        for key in ('demand_group', 'demand_hash', 'seed', 'episode', 'method',
                    'crc_trajectory_candidate', 'metadata_compatible'):
            if fresh[key] != old[key]:
                raise ValueError(f"previous archive identity changed: {old['path']} ({key})")


def preserve_previous_canonical(duplicates, previous_rows, previous_census):
    """Keep existing compact-cache sources when new paths sort before them."""
    excluded = {path for group in previous_census.get('verified_duplicate_groups', [])
                for path in group['files'][1:]}
    canonical = {row['path'] for row in previous_rows} - excluded
    anchored = []
    for group in duplicates:
        old_canonical = canonical.intersection(group['files'])
        if len(old_canonical) > 1:
            raise ValueError('supplement merged distinct previous canonical trajectories')
        files = group['files']
        if old_canonical:
            anchor = next(iter(old_canonical))
            files = [anchor] + [path for path in files if path != anchor]
        anchored.append({**group, 'files': files})
    return anchored


def collection_sources(collection, live_manifest, collection_manifests):
    pairs = [(collection, live_manifest)] if collection_manifests is None else collection_manifests
    if not pairs:
        raise ValueError('at least one collection manifest is required')
    sources = sorted({(str(Path(root).resolve()), str(Path(manifest).resolve()))
                      for root, manifest in pairs})
    if len({root for root, _ in sources}) != len(sources):
        raise ValueError('a collection root must have only one manifest')
    return [dict(collection_root=root, live_manifest=manifest) for root, manifest in sources]


def validate_collection(source, city, config, prepared):
    live = json.loads(Path(source['live_manifest']).read_text())
    configs = {row['command'][row['command'].index('--config') + 1]
               for row in live['runs'] if row['city'] == city}
    if not configs or live.get('input_hashes_by_city', {}).get(city) != prepared.input_hashes:
        raise ValueError('live collection input identity differs from previous pretraining')
    for path in configs:
        actual = load_config(Path(path))
        if (actual.inputs.network_name != city
                or validate_prepared_inputs(actual).input_hashes != prepared.input_hashes
                or physics_identity(actual.simulation) != physics_identity(config.simulation)
                or any(actual.resolved_snapshot()['simulation'][key] != config.resolved_snapshot()['simulation'][key]
                       for key in PHYSICAL)):
            raise ValueError('live collection physics/inputs differ from previous pretraining')


def scan_supplement_metadata(path, config, prepared, edge_ids, previous_by_path, sources, city):
    row = scan_metadata(path, config, prepared, edge_ids)
    if path in previous_by_path:
        # This field participates in immutable cache receipt identity, including
        # historical rows which were written with the original collection root.
        row['root'] = previous_by_path[path]['root']
    else:
        roots = [source['collection_root'] for source in sources
                 if Path(path).is_relative_to(Path(source['collection_root']) / city)]
        if len(roots) != 1:
            raise ValueError(f'archive has no unambiguous collection root: {path}')
        row['root'] = roots[0]
    return row


def prepare(city, previous_city_dir, output, *, collection=COLLECTION, live_manifest=LIVE_MANIFEST,
            collection_manifests=None):
    previous, output = Path(previous_city_dir).resolve(), Path(output).resolve()
    if output == previous or previous in output.parents:
        raise ValueError('supplement output must be isolated from the previous city run')
    previous_rows, previous_census, previous_holdout, seed = previous_partition(previous)
    config = load_config(previous / 'base_config.toml')
    if config.inputs.network_name != city:
        raise ValueError('previous configuration belongs to another city')
    prepared = validate_prepared_inputs(config)
    if previous_census['input_hashes'] != prepared.input_hashes:
        raise ValueError('previous frozen inputs differ from census')
    topology = load_traffic_topology(prepared)
    sources = collection_sources(collection, live_manifest, collection_manifests)
    for source in sources:
        validate_collection(source, city, config, prepared)
    frozen_path = output / 'census/frozen_candidates.json'
    old_paths = {r['path'] for r in previous_rows}
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
    else:
        collections = [freeze_committed(Path(source['collection_root']) / city) for source in sources]
        latest_paths = sorted({path for current in collections for path in current['paths']})
        frozen = dict(frozen_at=min(current['frozen_at'] for current in collections),
                      committed_bounds=(collections[0]['committed_bounds'] if len(sources) == 1 else
                                        {source['collection_root']: current['committed_bounds']
                                         for source, current in zip(sources, collections)}),
                      latest_committed_paths=latest_paths, paths=sorted(old_paths | set(latest_paths)))
        frozen.update(city=city, input_hashes=prepared.input_hashes,
                      previous_city_dir=str(previous), collection_sources=sources)
        if len(sources) == 1:
            frozen['collection_root'] = sources[0]['collection_root']
        write_json(frozen_path, frozen)
    if (frozen.get('city') != city or frozen.get('input_hashes') != prepared.input_hashes
            or frozen.get('previous_city_dir') != str(previous)
            or not old_paths <= set(frozen['paths'])
            or (frozen.get('collection_sources') != sources if 'collection_sources' in frozen else
                len(sources) != 1 or frozen.get('collection_root') != sources[0]['collection_root'])):
        raise ValueError('existing frozen supplement differs from requested anchor/inputs')
    for label, source in [('network_xml', prepared.network_xml), ('network_agents', prepared.network_agents_csv),
                          ('od_file', prepared.od_file), ('edge_registry', prepared.edge_registry)]:
        destination = output / 'inputs' / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != prepared.input_hashes[label]:
                raise ValueError('supplement input snapshot changed')
        else:
            shutil.copy2(source, destination)
    config = refreshed(config, inputs=replace(config.inputs, network_dir=output / 'inputs'),
                       run=replace(config.run, output_dir=output / 'world_model', seed=seed),
                       data=replace(config.data, archive_roots=tuple(sorted(
                           set(config.data.archive_roots) |
                           {Path(source['collection_root']) / city for source in sources}))))
    export_relative(config, output / 'base_config.toml')
    export_relative(config, output / 'pilot/run_config.toml')
    with ThreadPoolExecutor(max_workers=4) as pool:
        previous_by_path = {row['path']: row for row in previous_rows}
        rows = list(pool.map(lambda path: scan_supplement_metadata(path, config, prepared,
                            list(topology.network.edge_ids), previous_by_path, sources, city), frozen['paths']))
    verify_old_rows(rows, previous_rows)
    holdout = anchored_holdout(rows, previous_rows, previous_holdout, seed=seed)
    duplicates, unique = verify_duplicates(rows)
    duplicates = preserve_previous_canonical(duplicates, previous_rows, previous_census)
    census = dict(schema_version=1, city=city, frozen_at=frozen['frozen_at'],
                  input_hashes=prepared.input_hashes, committed_bounds=frozen['committed_bounds'],
                  files=len(rows), unique_trajectories=unique, verified_duplicate_groups=duplicates)
    write_json(output / 'census/archive_paths.json', frozen['paths'])
    write_json(output / 'census/file_rows.json', dict(rows=rows))
    write_json(output / 'census/census.json', census)
    excluded = {path for group in duplicates for path in group['files'][1:]}
    eligible = [row for row in rows if row['path'] not in excluded]
    train = representative_rows([r for r in eligible if r['demand_group'] not in holdout], 256, seed=seed)
    validation = representative_rows([r for r in eligible if r['demand_group'] in holdout], 48, seed=seed)
    if not train or not validation:
        raise ValueError('supplement requires nonempty disjoint train/validation partitions')
    write_json(output / 'pilot/split.json', dict(seed=seed, holdout_groups=sorted(holdout),
               train=train, validation=validation,
               train_demand_groups=len({r['demand_group'] for r in train}),
               validation_demand_groups=len({r['demand_group'] for r in validation}),
               scope='Old full-corpus assignments preserved; unseen groups use seeded 15% hash threshold.'))
    new_rows = [r for r in rows if r['path'] not in old_paths]
    # New-only census permits incremental cache building. The full census owns
    # cross-corpus deduplication, which must also be checked when merging caches.
    new_duplicates, new_unique = verify_duplicates(new_rows)
    write_json(output / 'new_census/archive_paths.json', [r['path'] for r in new_rows])
    write_json(output / 'new_census/file_rows.json', dict(rows=new_rows))
    write_json(output / 'new_census/census.json', {**census, 'files': len(new_rows),
               'unique_trajectories': new_unique, 'verified_duplicate_groups': new_duplicates})
    old_unique = previous_census['unique_trajectories']
    summary = dict(city=city, previous_city_dir=str(previous), previous_files=len(previous_rows),
                   previous_unique_trajectories=old_unique, new_files=len(new_rows),
                   added_unique_trajectories=unique-old_unique, frozen_files=len(rows), unique_trajectories=unique,
                   duplicates=len(rows)-unique, methods=dict(Counter(r['method'] for r in rows)),
                   previous_methods=dict(Counter(r['method'] for r in previous_rows)),
                   new_methods=dict(Counter(r['method'] for r in new_rows)),
                   latest_committed_files=len(frozen['latest_committed_paths']),
                   committed_bounds=frozen['committed_bounds'], heldout_demand_groups=len(holdout),
                   previous_demand_groups=len({r['demand_group'] for r in previous_rows}),
                   new_demand_groups=len({r['demand_group'] for r in rows}-{r['demand_group'] for r in previous_rows}),
                   previous_partition_preserved=True,
                   full_train_episodes=sum(r['demand_group'] not in holdout for r in eligible),
                   full_validation_episodes=sum(r['demand_group'] in holdout for r in eligible),
                   base_config=str(output / 'base_config.toml'), census_dir=str(output / 'census'),
                   new_census_dir=str(output / 'new_census'), pilot_dir=str(output / 'pilot'))
    write_json(output / 'preparation_summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city', required=True, choices=('coulommiers', 'montereau_fault_yonne'))
    parser.add_argument('--previous-city-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--collection-manifest', action='append', type=Path,
                        help='Repeat for each compatible collection; archive root is the manifest parent.')
    args = parser.parse_args(argv)
    sources = ([(manifest.parent, manifest) for manifest in args.collection_manifest]
               if args.collection_manifest else None)
    print(json.dumps(prepare(args.city, args.previous_city_dir, args.output_dir,
                             collection_manifests=sources), indent=2), flush=True)


if __name__ == '__main__':
    main()
