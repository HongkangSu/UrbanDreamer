"""Inventory compatible real trajectories without creating caches or training.

CRC member signatures select duplicate candidates only. Existing verified SHA
identities may be reused with an unchanged source stamp or complete member table;
remaining duplicate candidates are hashed over decompressed trajectory bytes.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_city_pretrain import PHYSICAL, DEMAND_HASH_KEYS, write_json
from urban_dreamer.learning.episode_archive import archive_physics_identity
from urban_dreamer.learning.offline_data import _source_config
from urban_dreamer.learning.pretrain_cache import file_stamp, member_signature
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.simulation_settings import physics_identity
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config

CITIES = ('ingolstadt_custom', 'coulommiers', 'montereau_fault_yonne')
CACHE_PATHS = {
    'ingolstadt_custom': ROOT/'outputs/ingolstadt_full_retrain_20261009/data/cache_manifest.json',
    **{c: ROOT/f'outputs/two_city_urbandreamer_pretrain_supplement_20261012/{c}/cache/cache_manifest.json' for c in CITIES[1:]}}


def decode_json(raw):
    value = np.load(io.BytesIO(raw), allow_pickle=False)
    return json.loads(value.tobytes() if value.ndim else value.item())


def array_shape(archive, member):
    with archive.open(member) as stream:
        version = np.lib.format.read_magic(stream)
        reader = getattr(np.lib.format, f'read_array_header_{version[0]}_{version[1]}')
        shape, _, _ = reader(stream)
        return tuple(shape)


def train_phase(metadata):
    """Online collection is training; explicit evaluation labels take precedence."""
    phase = metadata.get('phase')
    collection = metadata.get('collection_phase')
    if collection in {'evaluation', 'before_evaluation', 'after_evaluation'}:
        return False
    if phase is not None:
        return phase in {'train', 'online', 'training'}
    return collection in {'online', 'train', 'training'}


def stable_holdout(groups, forced, old_train=(), *, seed=42):
    # Threshold hashing is append-stable: adding new groups never reassigns old ones.
    if set(forced) & set(old_train):
        raise ValueError('old_demand_group_split_conflict')
    return set(forced) | {g for g in set(groups)-set(old_train) if
        int(hashlib.sha256(f'total-pretrain:{seed}:{g}'.encode()).hexdigest(), 16) / 2**256 < .15}


@lru_cache(maxsize=2048)
def inherited_source(directory, city, protocol, step):
    # Snapshot lookup depends on directory/city/protocol/grid, not episode identity.
    return _source_config(Path(directory)/'episode_probe.npz', dict(city=city,
        session_protocol=protocol, requested_step_seconds=step))['simulation']


def compatible_summary(summary, path, targets):
    if summary.get('source') != 'real_sumo':
        raise ValueError('not_real_sumo')
    meta = summary.get('metadata', {})
    city = meta.get('city')
    if city not in targets:
        raise ValueError('not_target_city')
    if not train_phase(meta):
        raise ValueError('not_training_partition')
    target = targets[city]
    if meta.get('input_hashes') != target['input_hashes'] or meta.get('edge_ids') != target['edge_ids']:
        raise ValueError('input_hash_or_edge_order_mismatch')
    if meta.get('session_protocol') != 'event' or meta.get('action_semantics') != 'branch-only':
        raise ValueError('action_protocol_mismatch')
    step = meta.get('requested_step_seconds', meta.get('step_seconds', meta.get('observation_step_seconds')))
    if step != 1. or meta.get('macro_step_seconds', 30.) != 30.:
        raise ValueError('observation_or_macro_grid_mismatch')
    embedded = meta.get('config') or meta.get('resolved_config')
    source = (embedded['simulation'] if isinstance(embedded, dict) else
              inherited_source(str(Path(path).parent), city, meta['session_protocol'], step))
    if archive_physics_identity(meta, simulation=source) != target['physics_identity']:
        raise ValueError('traffic_physics_mismatch')
    differences = [k for k in PHYSICAL if source.get(k) != target['physical'][k]]
    if differences:
        raise ValueError('physical_parameters_mismatch:' + ','.join(differences))
    demand = next((meta[k] for k in DEMAND_HASH_KEYS if meta.get(k)), None)
    if not demand:
        raise ValueError('missing_demand_identity')
    if meta.get('demand_law', meta.get('demand', {}).get('law')) not in {'independent_od_poisson', 'independent_poisson_per_od'}:
        raise ValueError('incompatible_demand_law')
    return dict(city=city, demand_group=demand, method=meta.get('method', 'urbandreamer' if meta.get('collection_phase') == 'online' else 'unknown'),
                episode=meta.get('episode'), population=meta.get('total_vehicle_count'),
                completed=summary.get('completed'), routing_mode=meta.get('routing_mode', source.get('routing_mode', 'next_hop')))


def streaming_trajectory_hash(path, expected_signature):
    """Exactly match read_hash_source trajectory SHA without decoding arrays/JSON."""
    path = Path(path)
    before = file_stamp(path)
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        members = sorted(archive.infolist(), key=lambda m: m.filename)
        if member_signature(members, trajectory_only=True) != expected_signature:
            raise ValueError('trajectory_changed_since_metadata')
        for member in members:
            if member.filename == 'summary_json.npy':
                continue
            digest.update(member.filename.encode() + b'\0')
            with archive.open(member) as stream:
                while chunk := stream.read(1024*1024):
                    digest.update(chunk)
            digest.update(b'\0')
    after = file_stamp(path)
    if after != before:
        # Hash the replacement too; lossless repacking is accepted only if SHA agrees.
        replacement = streaming_trajectory_hash(path, expected_signature)
        if replacement != digest.hexdigest():
            raise ValueError('nonlossless_replacement_during_hash')
    return digest.hexdigest()


def target_contracts():
    targets = {}
    for city in CITIES:
        config = load_config(ROOT/f'configs/{city}.toml')
        prepared = validate_prepared_inputs(config)
        topology = load_traffic_topology(prepared)
        snapshot = config.resolved_snapshot()['simulation']
        targets[city] = dict(input_hashes=prepared.input_hashes, edge_ids=list(topology.network.edge_ids),
                             physics_identity=physics_identity(config.simulation), physical={k:snapshot[k] for k in PHYSICAL})
    return targets


def load_known(targets):
    stamps, signatures, cache_rows, forced, old_train = {}, {}, [], defaultdict(set), defaultdict(set)
    for city, manifest_path in CACHE_PATHS.items():
        data = json.loads(manifest_path.read_text())
        contract = data['contract']
        for key in ('input_hashes', 'physics_identity', 'physical'):
            if contract[key] != targets[city][key]:
                raise ValueError(f'old_cache_contract_mismatch:{city}:{key}')
        for e in data['entries']:
            row = dict(e, city=city, crc_trajectory_candidate=e['census_trajectory_signature'],
                       metadata_compatible=True, cache_manifest=str(manifest_path), kind='verified_cache')
            if e['split'] == 'validation':
                forced[city].add(e['demand_group'])
            elif e['split'] == 'train':
                old_train[city].add(e['demand_group'])
            else:
                raise ValueError('invalid_old_cache_split')
            if e.get('source_stamp'):
                stamps[tuple(e['source_stamp'])] = row
            if e.get('source_member_signature'):
                signatures[e['source_member_signature']] = row
            cache_rows.append(row)
    for city in CITIES:
        if forced[city] & old_train[city]:
            raise ValueError(f'old_demand_group_split_conflict:{city}')
    return stamps, signatures, cache_rows, forced, old_train


def verify_cache(row):
    try:
        path = Path(row['cache_path'])
        with zipfile.ZipFile(path) as z:
            metadata = decode_json(z.read('metadata_json.npy'))
            shape = array_shape(z, 'action_counts.npy')
            if metadata['city'] != row['city'] or shape[0] != row['windows']:
                raise ValueError('cache_metadata_or_window_mismatch')
        return dict(row, path=str(path), source_stamp=file_stamp(path), verification='trusted_cache_manifest_plus_readable_header')
    except Exception as exc:
        return {'excluded':True, 'path':row['cache_path'], 'reason':'cache_unavailable:'+str(exc), 'city':row['city']}


def scan_one(item, targets, known_stamps, known_signatures):
    candidates = [item] + item.get('aliases', [])
    errors = []
    for candidate in candidates:
        path = Path(candidate['path'])
        try:
            stamp = file_stamp(path)
            known = known_stamps.get(tuple(stamp))
            if known:
                return dict(known, kind='raw', path=str(path), aliases=[x['path'] for x in candidates if x['path']!=str(path)],
                            source_stamp=stamp, verification='unchanged_verified_source_stamp')
            with zipfile.ZipFile(path) as z:
                members = z.infolist()
                names = [m.filename for m in members]
                if len(names) != len(set(names)) or 'summary_json.npy' not in names or any(not n.endswith('.npy') for n in names):
                    raise ValueError('not_raw_episode_schema')
                signature = member_signature(members)
                trajectory = member_signature(members, trajectory_only=True)
                known = known_signatures.get(signature)
                if known:
                    row = dict(known, verification='unchanged_complete_member_signature')
                else:
                    summary = decode_json(z.read('summary_json.npy'))
                    row = compatible_summary(summary, path, targets)
                    shape = array_shape(z, 'action.npy')
                    if len(shape)!=2 or shape[1]!=len(targets[row['city']]['edge_ids']):
                        raise ValueError('action_shape_mismatch')
                    row.update(windows=shape[0], metadata_compatible=True, verification='raw_metadata_and_header')
            after = file_stamp(path)
            if after != stamp:
                with zipfile.ZipFile(path) as changed:
                    if member_signature(changed.infolist()) != signature:
                        raise ValueError('nonlossless_replacement_during_metadata')
            return dict(row, kind='raw', path=str(path), aliases=[x['path'] for x in candidates if x['path']!=str(path)],
                        source_stamp=after, crc_trajectory_candidate=trajectory, source_member_signature=signature,
                        source_replaced_during_scan=after!=stamp, inventory_stamp=[item['dev'],item['ino'],item['size'],item['mtime_ns']],
                        root=item.get('root'))
        except Exception as exc:
            errors.append(str(exc))
            # Missing/moving aliases may recover. Metadata exclusions apply to same bytes.
            if isinstance(exc, ValueError) and not str(exc).startswith('nonlossless_'):
                break
    return dict(excluded=True, path=item['path'], reason=errors[-1] if errors else 'no_candidate', city_hint=item.get('city_hint'))


def jsonl_write(path, rows):
    with path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')


def run(args):
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    targets = target_contracts()
    stamps, signatures, cache_rows, forced, old_train = load_known(targets)
    write_json(output/'target_contracts.json', targets)
    inventory = [json.loads(line) for line in args.inventory.open()]
    fingerprint = hashlib.sha256(args.inventory.read_bytes()).hexdigest()
    protocol = dict(inventory_sha256=fingerprint, inventory_rows=len(inventory), targets=targets,
                    cache_manifests={c:str(p) for c,p in CACHE_PATHS.items()},
                    split='all old train/validation group assignments preserved; never-seen groups append-stable SHA threshold 15%',
                    routing_mode='not an exclusion; event branch-only action semantics required')
    protocol_path = output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('resume_protocol_mismatch')
    write_json(protocol_path, protocol)
    metadata_path = output/'metadata_rows.jsonl'
    if metadata_path.exists() and (output/'metadata_complete.json').exists():
        rows=[json.loads(line) for line in metadata_path.open()]
    else:
        rows=[]; started=time.time()
        if metadata_path.exists():
            for line in metadata_path.open():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: break
        done={r.get('input_path',r['path']) for r in rows}
        jobs=[('raw',x) for x in inventory if x['path'] not in done]+[('cache',x) for x in cache_rows if x['cache_path'] not in done]
        # Rewrite only complete checkpoint lines before resuming append.
        jsonl_write(metadata_path,rows)
        total=len(jobs)+len(rows)
        with ThreadPoolExecutor(max_workers=args.workers) as pool, metadata_path.open('a') as dest:
            futures={pool.submit(scan_one,x,targets,stamps,signatures) if kind=='raw' else pool.submit(verify_cache,x):(x['path'] if kind=='raw' else x['cache_path']) for kind,x in jobs}
            for future in as_completed(futures):
                row=future.result(); row['input_path']=futures[future];rows.append(row);dest.write(json.dumps(row,ensure_ascii=False)+'\n')
                if len(rows)%500==0:
                    dest.flush();progress=dict(stage='metadata',processed=len(rows),total=total,seconds=round(time.time()-started),
                        accepted=sum(not x.get('excluded') for x in rows))
                    write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
        write_json(output/'metadata_complete.json',dict(rows=len(rows),finished_utc=datetime.now(timezone.utc).isoformat()))
    accepted=[r for r in rows if not r.get('excluded')]
    excluded=[r for r in rows if r.get('excluded')]
    groups=defaultdict(list)
    for row in accepted:groups[(row['city'],row['crc_trajectory_candidate'])].append(row)
    # Existing SHA values are verified content provenance, never CRC-only identities.
    needs=[r for group in groups.values() if len(group)>1 for r in group if not r.get('trajectory_hash')]
    sha_path=output/'verified_new_hashes.jsonl'; sha_map={}
    if sha_path.exists():
        for line in sha_path.open():
            r=json.loads(line);sha_map[(r['path'],r['signature'])]=r['trajectory_hash']
    pending=[r for r in needs if (r['path'],r['crc_trajectory_candidate']) not in sha_map]
    write_json(output/'duplicate_workload.json',dict(compatible_rows=len(accepted),crc_buckets=len(groups),
               duplicate_buckets=sum(len(g)>1 for g in groups.values()),requires_full_hash=len(needs),pending_full_hash=len(pending)))
    print(json.dumps(dict(stage='duplicate_sha',pending=len(pending),crc_buckets=len(groups))),flush=True)
    with ThreadPoolExecutor(max_workers=args.hash_workers) as pool, sha_path.open('a') as log:
        futures={pool.submit(streaming_trajectory_hash,Path(r['path']),r['crc_trajectory_candidate']):r for r in pending}
        for i,f in enumerate(as_completed(futures),1):
            r=futures[f]
            try:value=f.result()
            except Exception as exc:
                write_json(output/'hash_failure.json',dict(path=r['path'],error=str(exc)))
                raise
            record=dict(path=r['path'],signature=r['crc_trajectory_candidate'],trajectory_hash=value)
            sha_map[(r['path'],r['crc_trajectory_candidate'])]=value;log.write(json.dumps(record)+'\n');log.flush()
            if i%100==0:
                progress=dict(stage='duplicate_sha',hashed=i,total=len(pending));write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    canonical=[];duplicates=[]
    for (city,signature), group in groups.items():
        buckets=defaultdict(list)
        for r in group:
            identity=r.get('trajectory_hash') or sha_map.get((r['path'],signature))
            # Different CRC signatures cannot be identical bytes. Singleton requires no full hash.
            key=identity if identity else 'unique_crc_bucket:'+signature
            r=dict(r,trajectory_hash=identity,identity_evidence='sha256' if identity else 'singleton_distinct_member_signature')
            buckets[key].append(r)
        for identity,members in buckets.items():
            if len({r['demand_group'] for r in members})!=1:
                raise ValueError('identical_trajectory_conflicting_demand_group')
            members.sort(key=lambda r:(r['kind']!='verified_cache', r['path']))
            row=dict(members[0]);row['equivalent_sources']=[dict(path=r['path'],kind=r['kind'],aliases=r.get('aliases',[])) for r in members]
            row['reusable_cache']=next((r['path'] for r in members if r['kind']=='verified_cache'),None)
            canonical.append(row)
            if len(members)>1:duplicates.append(dict(city=city,identity=identity,removed=len(members)-1,paths=[r['path'] for r in members]))
    summaries={}
    for city in CITIES:
        selected=[r for r in canonical if r['city']==city]
        demand={r['demand_group'] for r in selected};held=stable_holdout(demand,forced[city],old_train[city])
        for r in selected:r['split']='validation' if r['demand_group'] in held else 'train'
        jsonl_write(output/f'{city}_canonical.jsonl',selected)
        jsonl_write(output/f'{city}_train.jsonl',[r for r in selected if r['split']=='train'])
        jsonl_write(output/f'{city}_validation.jsonl',[r for r in selected if r['split']=='validation'])
        summaries[city]=dict(unique_trajectories=len(selected),train_trajectories=sum(r['split']=='train' for r in selected),
            validation_trajectories=sum(r['split']=='validation' for r in selected),independent_demand_groups=len(demand),
            train_demand_groups=len(demand-held),validation_demand_groups=len(demand&held),forced_old_holdout_groups=len(forced[city]),preserved_old_train_groups=len(old_train[city]),
            windows=sum(r['windows'] for r in selected),train_windows=sum(r['windows'] for r in selected if r['split']=='train'),
            validation_windows=sum(r['windows'] for r in selected if r['split']=='validation'),
            reusable_cache_trajectories=sum(bool(r['reusable_cache']) for r in selected),needs_cache_conversion=sum(not r['reusable_cache'] for r in selected),
            methods=dict(Counter(r['method'] for r in selected)),verified_duplicate_rows_removed=sum(g['removed'] for g in duplicates if g['city']==city))
    jsonl_write(output/'exclusions.jsonl',excluded);jsonl_write(output/'verified_duplicates.jsonl',duplicates)
    summary=dict(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),cities=summaries,
        inventory_raw_candidates=len(inventory),metadata_rows=len(rows),compatible_rows=len(accepted),excluded_rows=len(excluded),
        exclusion_reasons=dict(Counter(r['reason'] for r in excluded)),verification_reuse=dict(Counter(r.get('verification') for r in accepted)),
        full_new_duplicate_hashes=len(needs),unique_trajectories=sum(x['unique_trajectories'] for x in summaries.values()),
        limitations=['Compatible original bytes verified by old manifest provenance or metadata/headers; full-body SHA is required only for duplicate candidate groups.',
                    'Singleton distinct member signatures imply different bytes, without claiming their whole archives were fully decoded or CRC-tested.',
                    'The filesystem inventory is frozen; later completed episodes are not included.',
                    'Routing-mode differences are retained under shared event branch-only macro-action semantics.'])
    write_json(output/'summary.json',summary);write_json(output/'progress.json',dict(stage='complete',cities=summaries));print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,default=ROOT/'outputs/pretrain_data_audit_20261016/all_raw_inventory.jsonl')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/pretrain_data_audit_20261016/total_census')
    p.add_argument('--workers',type=int,default=16);p.add_argument('--hash-workers',type=int,default=8)
    run(p.parse_args())
if __name__=='__main__':main()
