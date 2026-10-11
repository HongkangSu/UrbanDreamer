"""Freeze committed current-city SUMO data for codec and full-corpus pretraining.

This prepares files only. Collectors and lossless archive compactors may continue.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.compact_episode_archives import committed_episode
from tools.run_complete_pretrain import export_run_config
from urban_dreamer.learning.episode_archive import archive_physics_identity
from urban_dreamer.learning.offline_data import _source_config
from urban_dreamer.learning.pretrain_cache import file_stamp, member_signature, read_hash_source
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.simulation_settings import physics_identity
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import DataConfig, _semantic_fingerprint, load_config

COLLECTION = ROOT / 'outputs/city_baselines_v5_1500_noprior_penalty_20261009'
LIVE_MANIFEST = ROOT / 'outputs/city_baselines_paper_reward_20261010/manifest.json'
PHYSICAL = ('vehicle_max_speed_mps', 'incident_edge', 'incident_start_seconds',
            'incident_end_seconds', 'incident_speed_mps', 'sumo_extra_args', 'demand_duration_seconds')
DEMAND_HASH_KEYS = ('demand_sha256', 'demand_hash', 'demand_realization_sha256', 'demand_realization_hash')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def freeze_committed(city_root, *, cursor_reader=committed_episode):
    """Freeze explicit paths against durable checkpoints, never filename maxima."""
    bounds, paths = {}, []
    for directory in sorted(city_root.iterdir()):
        if not directory.is_dir():
            continue
        archives = sorted((directory / 'episodes/train').glob('episode_*.npz'))
        if not archives:
            continue
        bound = cursor_reader(dict(method=directory.name, output_dir=str(directory)))
        selected = [p.resolve() for p in archives if int(p.stem.removeprefix('episode_')) <= bound]
        bounds[directory.name] = dict(committed_episode=bound, files=len(selected))
        paths.extend(map(str, selected))
    if not paths:
        raise ValueError('no checkpoint-committed training archives')
    return dict(frozen_at=datetime.now(timezone.utc).isoformat(), committed_bounds=bounds, paths=paths)


def scan_metadata(path, config, prepared, edge_ids):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        if not names or len(names) != len(set(names)) or any(not n.endswith('.npy') for n in names):
            raise ValueError(f'{path}: invalid NPZ member table')
        raw = np.load(io.BytesIO(archive.read('summary_json.npy')), allow_pickle=False)
        summary = json.loads(raw.tobytes() if raw.ndim else raw.item())
    meta = summary['metadata']
    if (meta.get('city') != config.inputs.network_name or meta.get('phase') != 'train'
            or summary.get('source') != 'real_sumo' or meta.get('input_hashes') != prepared.input_hashes
            or meta.get('edge_ids') != edge_ids or meta.get('session_protocol') != 'event'
            or meta.get('action_semantics') != 'branch-only'):
        raise ValueError(f'{path}: archive city/input/partition/protocol mismatch')
    step = meta.get('requested_step_seconds', meta.get('step_seconds', meta.get('observation_step_seconds')))
    if step != 1. or meta.get('macro_step_seconds', 30.) != 30.:
        raise ValueError(f'{path}: incompatible observation/macro grid')
    source = _source_config(path, meta)['simulation']
    target = config.resolved_snapshot()['simulation']
    if (archive_physics_identity(meta, simulation=source) != physics_identity(config.simulation)
            or any(source.get(key) != target[key] for key in PHYSICAL)):
        raise ValueError(f'{path}: physical/demand configuration mismatch')
    demand = next((meta[key] for key in DEMAND_HASH_KEYS if meta.get(key)), None)
    if not demand or meta.get('demand_law', meta.get('demand', {}).get('law')) not in {
            'independent_od_poisson', 'independent_poisson_per_od'}:
        raise ValueError(f'{path}: missing demand realization identity/law')
    if int(path.stem.removeprefix('episode_')) != meta['episode']:
        raise ValueError(f'{path}: episode filename differs from metadata')
    method = path.parent.parent.parent.name
    return dict(path=str(path), root=str(COLLECTION), method=method, episode=meta['episode'],
                seed=meta['seed'], demand_hash=demand, demand_group=demand, metadata_compatible=True,
                source_stamp=file_stamp(path), crc_trajectory_candidate=member_signature(members, trajectory_only=True))


def verify_duplicates(rows, *, reader=read_hash_source):
    """CRC candidates reduce I/O; only full SHA256 establishes duplicates."""
    candidates = defaultdict(list)
    for row in rows:
        candidates[row['crc_trajectory_candidate']].append(row)
    verified, duplicate_count = [], 0
    for group in candidates.values():
        if len(group) < 2:
            continue
        hashes = defaultdict(list)
        for row in sorted(group, key=lambda r: r['path']):
            _, identity = reader(Path(row['path']))
            if identity['census_trajectory_signature'] != row['crc_trajectory_candidate']:
                raise ValueError(f"frozen trajectory changed: {row['path']}")
            hashes[identity['trajectory_hash']].append(row)
        for trajectory, members in hashes.items():
            if len(members) < 2:
                continue
            if len({row['demand_group'] for row in members}) != 1:
                raise ValueError('identical trajectory has conflicting demand identities')
            files = [row['path'] for row in members]
            verified.append(dict(trajectory_hash=trajectory, files=files))
            duplicate_count += len(files) - 1
    return verified, len(rows) - duplicate_count


def split_groups(rows, *, seed=42):
    groups = sorted({row['demand_group'] for row in rows},
                    key=lambda group: hashlib.sha256(f'city-pretrain:{seed}:{group}'.encode()).hexdigest())
    if len(groups) < 2:
        raise ValueError('at least two independent demand realizations are required')
    count = min(len(groups) - 1, max(1, round(.15 * len(groups))))
    return set(groups[:count])


def representative_rows(rows, limit, *, seed=42):
    """Round-robin across methods without changing the whole-demand split."""
    methods = defaultdict(list)
    for row in sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row['path']}".encode()).hexdigest()):
        methods[row['method']].append(row)
    selected = []
    while methods and len(selected) < limit:
        for method in sorted(list(methods)):
            selected.append(methods[method].pop())
            if not methods[method]:
                del methods[method]
            if len(selected) == limit:
                break
    return selected


def refreshed(config, **changes):
    config = replace(config, **changes)
    return replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))


def export_relative(config, path):
    export_run_config(config, path)
    relative = os.path.relpath(config.inputs.network_dir, path.parent)
    text = path.read_text().replace('network_dir = ' + json.dumps(str(config.inputs.network_dir)),
                                    'network_dir = ' + json.dumps(relative))
    path.write_text(text)
    if load_config(path).resolved_snapshot() != config.resolved_snapshot():
        raise ValueError('relative exported configuration differs from frozen semantics')


def prepare(city, output):
    output = output.resolve()
    config = load_config(ROOT / f'configs/{city}_offline.toml')
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    live = json.loads(LIVE_MANIFEST.read_text())
    live_configs = {row['command'][row['command'].index('--config') + 1]
                    for row in live['runs'] if row['city'] == city}
    if not live_configs or live['input_hashes_by_city'][city] != prepared.input_hashes:
        raise ValueError('current prepared inputs differ from running collection')
    for path in live_configs:
        actual = load_config(Path(path))
        if (validate_prepared_inputs(actual).input_hashes != prepared.input_hashes
                or physics_identity(actual.simulation) != physics_identity(config.simulation)
                or any(actual.resolved_snapshot()['simulation'][k] != config.resolved_snapshot()['simulation'][k]
                       for k in PHYSICAL)):
            raise ValueError('live collection config differs from offline physics/inputs')
    frozen_path = output / 'census/frozen_candidates.json'
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
    else:
        frozen = freeze_committed(COLLECTION / city)
        frozen.update(city=city, input_hashes=prepared.input_hashes, collection_root=str(COLLECTION))
        write_json(frozen_path, frozen)
    if frozen.get('city') != city or frozen.get('input_hashes') != prepared.input_hashes:
        raise ValueError('existing frozen membership differs from requested city/inputs')
    for label, source in [('network_xml', prepared.network_xml), ('network_agents', prepared.network_agents_csv),
                          ('od_file', prepared.od_file), ('edge_registry', prepared.edge_registry)]:
        destination = output / 'inputs' / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != prepared.input_hashes[label]:
                raise ValueError('frozen input snapshot changed')
        else:
            shutil.copy2(source, destination)
    config = refreshed(config, inputs=replace(config.inputs, network_dir=output / 'inputs'),
                       run=replace(config.run, output_dir=output / 'world_model', seed=42),
                       codec=replace(config.codec, batch_size=4),
                       data=DataConfig(archive_roots=(COLLECTION / city,), cache_episodes=2))
    export_relative(config, output / 'base_config.toml')
    export_relative(config, output / 'pilot/run_config.toml')
    census_path = output / 'census/census.json'
    if census_path.exists():
        census = json.loads(census_path.read_text())
        rows = json.loads((output / 'census/file_rows.json').read_text())['rows']
        if {r['path'] for r in rows} != set(frozen['paths']):
            raise ValueError('saved census membership differs from frozen paths')
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = pool.map(lambda path: scan_metadata(path, config, prepared, list(topology.network.edge_ids)), frozen['paths'])
            for index, row in enumerate(results, 1):
                rows.append(row)
                if index % 500 == 0:
                    print(json.dumps(dict(city=city, scanned=index, total=len(frozen['paths']))), flush=True)
        duplicates, unique = verify_duplicates(rows)
        census = dict(schema_version=1, city=city, frozen_at=frozen['frozen_at'],
                      input_hashes=prepared.input_hashes, committed_bounds=frozen['committed_bounds'],
                      files=len(rows), unique_trajectories=unique, verified_duplicate_groups=duplicates)
        write_json(output / 'census/archive_paths.json', frozen['paths'])
        write_json(output / 'census/file_rows.json', dict(rows=rows))
        write_json(census_path, census)
    holdout = split_groups(rows)
    noncanonical = {p for group in census['verified_duplicate_groups'] for p in group['files'][1:]}
    eligible = [r for r in rows if r['path'] not in noncanonical]
    train = representative_rows([r for r in eligible if r['demand_group'] not in holdout], 256)
    validation = representative_rows([r for r in eligible if r['demand_group'] in holdout], 48)
    split = dict(seed=42, holdout_groups=sorted(holdout), train=train, validation=validation,
                 train_demand_groups=len({r['demand_group'] for r in train}),
                 validation_demand_groups=len({r['demand_group'] for r in validation}),
                 scope='Representative codec pilot; full corpus cache retains every unique trajectory and full heldout split.')
    write_json(output / 'pilot/split.json', split)
    summary = dict(city=city, frozen_files=len(rows), unique_trajectories=census['unique_trajectories'],
                   duplicates=len(rows) - census['unique_trajectories'], methods=dict(Counter(r['method'] for r in rows)),
                   committed_bounds=frozen['committed_bounds'], codec_train_episodes=len(train),
                   codec_validation_episodes=len(validation), heldout_demand_groups=len(holdout),
                   full_train_episodes=sum(r['demand_group'] not in holdout for r in eligible),
                   full_validation_episodes=sum(r['demand_group'] in holdout for r in eligible),
                   base_config=str(output / 'base_config.toml'), census_dir=str(output / 'census'),
                   pilot_dir=str(output / 'pilot'))
    write_json(output / 'preparation_summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city', required=True, choices=('coulommiers', 'montereau_fault_yonne'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.city, args.output_dir), indent=2), flush=True)


if __name__ == '__main__':
    main()
