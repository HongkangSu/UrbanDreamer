#!/usr/bin/env python3
"""Run a bounded, seeded SUMO baseline comparison against a source snapshot."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--method', choices=['dreamer', 'mbpo'], required=True)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--vehicle-limit', type=int, default=50)
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('throughput_trainer', source / 'tools/train_toy_baseline.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    from urban_dreamer.learning.baselines import METHODS, common
    import torch

    measurements, pending = [], {}
    original_collect = common.EpisodeRunner.run_episode
    method_class = METHODS[args.method]
    original_update = method_class.on_episode

    def collect(self, *a, **kw):
        started = time.monotonic()
        result = original_collect(self, *a, **kw)
        if kw.get('archive_metadata', {}).get('phase') == 'train':
            pending['collection_seconds'] = time.monotonic() - started
        return result

    def update(self, *a, **kw):
        started = time.monotonic()
        result = original_update(self, *a, **kw)
        measurements.append(dict(episode=kw['episode'],
            collection_seconds=pending.pop('collection_seconds'),
            learning_seconds=time.monotonic() - started))
        return result

    common.EpisodeRunner.run_episode = collect
    method_class.on_episode = update
    trainer.main(['--method', args.method, '--config', str(source / 'configs/toy_cross_bottleneck.toml'),
        '--output-dir', str(args.output_dir), '--episodes', str(args.episodes),
        '--vehicle-limit', str(args.vehicle_limit), '--device', 'cpu', '--threads', '1',
        '--session', 'event', '--backend', 'libsumo', '--eval-every', '1000', '--eval-seeds', '1',
        '--skip-initial-evaluation', '--save-episodes', '--sp-prior-weight', '0'])
    checkpoint = torch.load(args.output_dir / 'latest.pt', map_location='cpu', weights_only=False)
    digest = hashlib.sha256()

    def tensors(value):
        if isinstance(value, torch.Tensor):
            digest.update(value.detach().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                if key not in ('model_rows', 'seed_buffer', 'sequence_buffer', 'window_buffer'):
                    tensors(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                tensors(item)

    tensors(checkpoint['method_state'])
    report = dict(source=str(source), method=args.method, vehicle_limit=args.vehicle_limit,
                  measurements=measurements, tensor_state_sha256=digest.hexdigest(),
                  note='Bounded CPU Toy benchmark with unchanged update counts; timing excludes evaluation/checkpoints.')
    (args.output_dir / 'throughput.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
