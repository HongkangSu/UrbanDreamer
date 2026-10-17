"""Preserve available argmax checkpoint weights, never inferred historical weights.

CPU-only sidecar. Version directories plus an atomic current symlink publish a
checkpoint/metadata pair together. Original checkpoints and training are untouched.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import torch

CITIES = ('ingolstadt_custom', 'coulommiers', 'montereau_fault_yonne')


def stamp(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def evaluation_candidate(payload, protocol):
    if protocol.get('evaluation', {}).get('policy') != 'argmax':
        raise ValueError('evaluation protocol is not argmax')
    state = payload['training_state']
    progress = state['metrics']['online_pilot_progress']
    count = len(progress['online'])
    if int(state['metrics']['online_episodes_completed']) != count:
        raise ValueError('checkpoint weights/progress episode mismatch')
    if payload['phase'] != 'completed' and payload['next_unit'] != count:
        raise ValueError('checkpoint cursor/progress mismatch')
    if payload['config_fingerprint'] != protocol['runtime_config_fingerprint']:
        raise ValueError('checkpoint/runtime configuration mismatch')
    rows = None
    if count == 0 and len(progress.get('before', [])) == 3:
        rows = progress['before']
    for milestone in progress.get('milestones', []):
        if milestone['online_episodes'] == count and milestone['phase'] == 'completed':
            rows = milestone['evaluations']
    if progress['phase'] == 'completed' and payload['phase'] == 'completed':
        rows = progress.get('after', [])
    if rows is None or len(rows) != 3:
        raise ValueError('no complete evaluation for this checkpoint episode')
    expected_indices = protocol['paired_evaluation_indices']
    if sorted(x['episode_index'] for x in rows) != sorted(expected_indices):
        raise ValueError('evaluation demand indices differ')
    before_hashes = {x['episode_index']: x['demand_hash'] for x in progress['before']}
    scores = []
    for row in rows:
        # complete denotes all vehicles arrived, not whether evaluation terminated.
        if not row.get('episode_terminal', False):
            raise ValueError('evaluation episode not terminal')
        if row['demand_hash'] != before_hashes.get(row['episode_index']):
            raise ValueError('evaluation demand hash differs from paired baseline')
        metrics = row['metrics']
        if metrics.get('evaluation_demand_seed') != protocol['evaluation_demand_seed']:
            raise ValueError('evaluation demand seed differs')
        score = float(metrics['evaluation_selection_score'])
        if not math.isfinite(score):
            raise ValueError('nonfinite score')
        scores.append(score)
    return dict(episode=count, mean_selection_score=sum(scores)/len(scores),
                evaluations=rows, evaluation_protocol=protocol['evaluation'],
                evaluation_demand_seed=protocol['evaluation_demand_seed'],
                demand_hashes=before_hashes, config_fingerprint=payload['config_fingerprint'],
                input_hash=payload['input_hash'], selection='best among actually available evaluated weights; not historical metric best')


def publish_open_checkpoint(handle, source_stamp, output, metadata):
    """Copy the same open inode that was loaded; atomic source replacement is safe."""
    output.mkdir(parents=True, exist_ok=True)
    generations = output / 'generations'; generations.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.candidate-', dir=generations))
    digest = hashlib.sha256()
    try:
        handle.seek(0)
        with (temporary / 'checkpoint.pt').open('wb') as target:
            while block := handle.read(8 * 1024 * 1024):
                target.write(block); digest.update(block)
            target.flush(); os.fsync(target.fileno())
        if stamp(os.fstat(handle.fileno())) != source_stamp:
            raise ValueError('checkpoint inode mutated while copying')
        metadata = dict(metadata, checkpoint_sha256=digest.hexdigest(), source_stamp=source_stamp,
                        preserved_unix=time.time())
        with (temporary / 'metadata.json').open('w') as target:
            json.dump(metadata, target, indent=2); target.write('\n'); target.flush(); os.fsync(target.fileno())
        final = generations / f"episode_{metadata['episode']:06d}_{digest.hexdigest()}"
        if final.exists():
            (temporary/'checkpoint.pt').unlink(); (temporary/'metadata.json').unlink(); temporary.rmdir()
        else:
            temporary.rename(final)
        link = output / '.current.next'
        link.unlink(missing_ok=True)
        link.symlink_to(final.relative_to(output), target_is_directory=True)
        os.replace(link, output/'current')
        for name, relative in [('best_available_argmax.pt','current/checkpoint.pt'),
                               ('best_available_argmax.json','current/metadata.json')]:
            alias = output/name
            if not alias.is_symlink():
                if alias.exists(): raise ValueError(f'refusing to replace existing non-symlink {alias}')
                alias.symlink_to(relative)
        fd=os.open(output,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        return metadata
    finally:
        if temporary.exists():
            for child in temporary.iterdir(): child.unlink()
            temporary.rmdir()


def inspect_and_preserve(checkpoint, protocol, output):
    with checkpoint.open('rb') as handle:
        before = stamp(os.fstat(handle.fileno()))
        # mmap leaves tensor storage on disk; only CPU metadata is needed.
        # /proc/self/fd pins the opened inode even if latest.pt is replaced.
        payload = torch.load(f'/proc/self/fd/{handle.fileno()}',
                             map_location='cpu', weights_only=True, mmap=True)
        if stamp(os.fstat(handle.fileno())) != before:
            raise ValueError('checkpoint inode mutated during load')
        candidate = evaluation_candidate(payload, protocol)
        candidate['source_checkpoint'] = str(checkpoint.resolve())
        current = output/'current/metadata.json'
        if current.exists():
            old = json.loads(current.read_text())
            for key in ['evaluation_protocol','evaluation_demand_seed','input_hash','config_fingerprint']:
                if old[key] != candidate[key]: raise ValueError(f'selection contract differs: {key}')
            if old['demand_hashes'] != {str(k):v for k,v in candidate['demand_hashes'].items()}:
                raise ValueError('selection demand hashes differ')
            if candidate['mean_selection_score'] <= old['mean_selection_score']:
                return dict(status='not_improved', episode=candidate['episode'], source_stamp=before)
        saved = publish_open_checkpoint(handle,before,output,candidate)
        return dict(status='saved', episode=saved['episode'], mean_selection_score=saved['mean_selection_score'],
                    sha256=saved['checkpoint_sha256'], source_stamp=before)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--interval',type=float,default=8)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    lock = (args.output/'.watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    seen={}
    while True:
        for city in CITIES:
            online=args.run_root/city/'online';checkpoint=online/'checkpoints/latest.pt'
            now=None
            try:
                now=stamp(checkpoint.stat())
                if seen.get(city)==now:continue
                retained=args.output/city/'current/metadata.json'
                if retained.exists() and tuple(json.loads(retained.read_text()).get('source_stamp',[]))==now:
                    seen[city]=now
                    continue
                protocol=json.loads((online/'pilot_protocol.json').read_text())
                # Cheap hint only: avoid unpickling replay every training episode.
                # Never mark the inode seen here: progress may lag atomic save.
                preview=json.loads((online/'pilot_progress.json').read_text())
                count=len(preview['online'])
                eligible=(count==0 and len(preview.get('before',[]))==3) or (
                    preview['phase']=='completed' and len(preview.get('after',[]))==3) or any(
                    m['online_episodes']==count and m['phase']=='completed' and len(m['evaluations'])==3
                    for m in preview.get('milestones',[]))
                if not eligible:
                    if args.once:
                        print(json.dumps(dict(city=city,status='waiting_for_matching_completed_evaluation',episode=count)),flush=True)
                    continue
                result=inspect_and_preserve(checkpoint,protocol,args.output/city)
                seen[city]=tuple(result.pop('source_stamp'))
            except (ValueError,KeyError,OSError,RuntimeError) as exc:
                result=dict(status='not_eligible_or_retry',reason=str(exc))
                if isinstance(exc,(ValueError,KeyError)) and now is not None: seen[city]=now
            record=dict(city=city,unix=time.time(),**result)
            print(json.dumps(record),flush=True)
            with (args.output/'watcher_events.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
        if args.once:break
        time.sleep(max(5,args.interval))


if __name__=='__main__':main()
