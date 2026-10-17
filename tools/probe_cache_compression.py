"""Bounded lossless compact-cache compression probe; never mutates sources."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from urban_dreamer.learning.pretrain_cache import read_hash_source, compact_arrays, boundary_vehicles


def hashes(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            digest = hashlib.sha256()
            with archive.open(name) as member:
                while chunk := member.read(1024 * 1024):
                    digest.update(chunk)
            result[name] = digest.hexdigest()
    return result


def repack(source, target):
    wall, cpu = time.monotonic(), time.process_time()
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_LZMA) as new:
        types = sorted({member.compress_type for member in old.infolist()})
        for member in old.infolist():
            with old.open(member) as src, new.open(member.filename, 'w', force_zip64=True) as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
    timing = dict(compress_wall_seconds=time.monotonic()-wall, compress_cpu_seconds=time.process_time()-cpu)
    start = time.monotonic()
    expected = hashes(source)
    old_read_seconds = time.monotonic()-start
    start = time.monotonic()
    assert hashes(target) == expected
    result = dict(source_bytes=source.stat().st_size, lzma_bytes=target.stat().st_size,
                  original_zip_methods=types, member_bytes_equal=True, members=len(expected),
                  source_read_seconds=old_read_seconds, lzma_read_seconds=time.monotonic()-start, **timing)
    result['saved_bytes'] = result['source_bytes']-result['lzma_bytes']
    result['saved_fraction'] = result['saved_bytes']/result['source_bytes']
    return result


def check_raw(data, arrays):
    for field in ('flow','time','action_counts','departure_counts','active_vehicle_seconds','terminated','truncated'):
        assert np.array_equal(arrays[field], data[field], equal_nan=True)
    for frame, state in enumerate(data['states']):
        wanted = [v for v in state['vehicles'] if v['departed'] and not v['arrived']]
        restored = boundary_vehicles(arrays, frame)
        assert len(wanted) == len(restored)
        for old, new in zip(wanted, restored):
            for key, value in new.items():
                default = False if key in ('in_junction', 'pending_decision') else (0. if key in ('junction_remaining_distance', 'junction_speed') else None)
                original = old.get(key, default)
                if key == 'junction_connection' and original is not None:
                    original = tuple(original)
                assert original == value, (frame, key, original, value)
    for i, event in enumerate(data['decisions']):
        assert str(arrays['vehicle_ids'][arrays['event_vehicle'][i]]) == event['vehicle_id']
        for key in ('window_index','event_index','current_edge','next_edge','destination_edge','decision_time'):
            assert arrays['event_'+key][i] == event[key]
    return dict(traffic_arrays_equal=True, all_active_boundaries_equal=True, all_events_equal=True,
                frames=len(data['states']), events=len(data['decisions']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/pretrain_storage_migration_20261016/compression_probe')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    corpus = json.loads((ROOT/'outputs/pretrain_data_audit_20261016/cached_corpus.json').read_text())
    inventory = {json.loads(line)['path'] for line in (ROOT/'outputs/pretrain_data_audit_20261016/all_raw_inventory.jsonl').open()}
    raw = {city: [] for city in corpus}
    for line in (ROOT/'outputs/pretrain_data_audit_20261016/total_census/metadata_rows.jsonl').open():
        row = json.loads(line)
        if row.get('city') in raw and row.get('metadata_compatible') and row['path'] in inventory and len(raw[row['city']]) < 2:
            raw[row['city']].append(row['path'])
    results = []
    def save():
        (args.output/'results.json').write_text(json.dumps(dict(status='running', rows=results), indent=2)+'\n')
    for city, info in corpus.items():
        entries = json.loads(Path(info['manifest']).read_text())['entries']
        for index in (0, len(entries)//2):
            source = Path(entries[index]['cache_path'])
            target = args.output/f'{city}_cache_{index}.npz'
            item = dict(city=city, kind='existing_cache', source=str(source), **repack(source,target))
            item['peak_extra_disk_bytes'] = target.stat().st_size
            target.unlink()
            results.append(item)
            save()
            print(json.dumps(item), flush=True)
        for index, name in enumerate(raw[city]):
            source = Path(name)
            start, cpu = time.monotonic(), time.process_time()
            data, identity = read_hash_source(source)
            arrays = compact_arrays(data)
            preserved = check_raw(data, arrays)
            conversion = dict(convert_wall_seconds=time.monotonic()-start, convert_cpu_seconds=time.process_time()-cpu)
            temporary = args.output/f'{city}_raw_{index}_deflate.npz'
            target = args.output/f'{city}_raw_{index}_lzma.npz'
            start, cpu = time.monotonic(), time.process_time()
            np.savez_compressed(temporary, **arrays)
            deflate = dict(deflate_wall_seconds=time.monotonic()-start, deflate_cpu_seconds=time.process_time()-cpu)
            item = dict(city=city, kind='new_raw_conversion', source=name, raw_bytes=source.stat().st_size,
                        identity=identity, preservation=preserved, **conversion, **deflate, **repack(temporary,target))
            with zipfile.ZipFile(source) as raw_archive:
                item['raw_zip_methods'] = sorted({m.compress_type for m in raw_archive.infolist()})
            item['raw_to_lzma_saved_fraction'] = 1-item['lzma_bytes']/item['raw_bytes']
            item['peak_extra_disk_bytes'] = temporary.stat().st_size+target.stat().st_size
            for compressed in (temporary,target):
                with np.load(compressed, allow_pickle=False) as packed:
                    assert set(packed.files) == set(arrays)
                    for key, value in arrays.items():
                        assert np.array_equal(packed[key],value)
            item['all_compact_arrays_roundtrip_equal'] = True
            temporary.unlink()
            target.unlink()
            results.append(item)
            save()
            print(json.dumps(item), flush=True)
    summary = dict(status='complete', rows=results, single_process=True, max_extra_disk_bytes=max(x['peak_extra_disk_bytes'] for x in results),
                   source_files_modified=False, temporary_probe_archives_removed=True)
    (args.output/'results.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__ == '__main__':
    main()
