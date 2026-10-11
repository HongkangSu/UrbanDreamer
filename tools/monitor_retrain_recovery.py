"""Verify a sealed user-authorized data expansion before releasing retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_full_retrain import atomic_json
from urban_dreamer.learning.offline_data import _identity


def bind_expanded_groups(binding, manifest):
    """Retain every old assignment; derive new assignments by the same frozen rule."""
    if binding.get('identity') != _identity({k: v for k, v in binding.items() if k != 'identity'}):
        raise ValueError('initial archived calibration binding changed')
    result = dict(binding)
    assignments = dict(binding['original_demand_groups'])
    added = 0
    for row in manifest['sources']:
        group, split = row['demand_group'], row['split']
        if group in assignments:
            if assignments[group] != split:
                raise ValueError('expanded data changed a preserved demand partition')
        else:
            fraction = int(hashlib.sha256(('full-retrain-v1:' + group).encode()).hexdigest()[:8], 16) / 2**32
            expected = 'validation' if fraction < .15 else 'train'
            if expected != split:
                raise ValueError('new demand partition differs from deterministic holdout rule')
            assignments[group] = split
            added += 1
    result['original_demand_groups'] = assignments
    result['authorized_expansion'] = dict(revision=manifest['expansion']['revision'],
        cache_manifest_identity=manifest['identity'], added_demand_groups=added,
        previous_binding_identity=binding['identity'])
    result.pop('identity')
    result['identity'] = _identity(result)
    return result


def process_alive(record):
    try:
        pid = record['pid']
        raw = Path(f'/proc/{pid}/stat').read_text()
        command = Path(f'/proc/{pid}/cmdline').read_bytes()
        return raw.rsplit(')', 1)[1].split()[0] != 'Z' and b'recover_full_retrain_data.py' in command
    except (FileNotFoundError, KeyError):
        return False


def run(args):
    data = args.run_dir / 'data'
    status_path = data / 'status.json'
    def status(kind, **values):
        atomic_json(status_path, dict(status=kind, stage='authorized_data_recovery_expansion',
            utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), **values))
    try:
        while True:
            record = json.loads((data / 'recovery_revision_process.json').read_text())
            progress_path = data / 'recovery_revision_progress.json'
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            manifest_path = data / 'cache_manifest.json'
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            complete = (manifest.get('status') == 'complete'
                and manifest.get('expansion', {}).get('status') == 'complete'
                and manifest.get('recovery', {}).get('raw_source_snapshot') == 'hardlink')
            if complete:
                status('verifying', pid=record['pid'], progress=progress)
                from tools.run_complete_pretrain import calibrated_codec_prerequisites
                from urban_dreamer.workflow.config import load_config
                import torch
                torch.set_num_threads(2)
                binding = bind_expanded_groups(json.loads(args.initial_binding.read_text()), manifest)
                if args.expanded_binding.exists() and json.loads(args.expanded_binding.read_text()) != binding:
                    raise ValueError('expanded calibration binding changed')
                atomic_json(args.expanded_binding, binding)
                config = load_config(args.config)
                check = calibrated_codec_prerequisites(args.calibrated_codec, manifest_path, config,
                    binding_path=args.expanded_binding, expected_train=manifest['train_episodes'],
                    expected_validation=manifest['validation_episodes'])
                if not check['ready']:
                    raise ValueError('sealed expanded data prerequisites not ready')
                atomic_json(data / 'calibrated_codec_prerequisites.json', check['proof'])
                status('complete', manifest_identity=manifest['identity'],
                    train_episodes=manifest['train_episodes'], validation_episodes=manifest['validation_episodes'],
                    archived_calibration_binding=True)
                return
            if not process_alive(record):
                log = data / 'recovery_revision_build.log'
                tail = log.read_text(errors='replace')[-6000:] if log.exists() else ''
                raise RuntimeError('data expansion ended before complete sealed manifest: ' + tail)
            status('building', pid=record['pid'], progress=progress)
            time.sleep(10)
    except BaseException as error:
        status('failed', error=str(error), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'config', 'calibrated-codec', 'initial-binding', 'expanded-binding'):
        parser.add_argument('--' + name, required=True, type=lambda value: Path(value).resolve())
    run(parser.parse_args())


if __name__ == '__main__':
    main()
