"""CPU-only checkpoint preparation A/B, without changing the source checkpoint."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, fields
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from urban_dreamer.learning.action_events import ActionEvent
from urban_dreamer.learning.replay import MacroTransition
from urban_dreamer.workflow.checkpoint import _normalize_plain_value


def reference_normalize(value):
    """Previous scalar-last recursion, for valid checkpoint payloads."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('object array')
        return torch.as_tensor(value).detach().cpu()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [reference_normalize(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise ValueError('invalid key')
            result[key] = reference_normalize(item)
        return result
    raise ValueError('unsupported value')


def reference_snapshot(row):
    result = {field.name: deepcopy(getattr(row, field.name)) for field in fields(row)}
    result['vehicles'] = list(result['vehicles'])
    if row.action_events is not None:
        result['action_events'] = [asdict(event) for event in row.action_events]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    torch.set_num_threads(1)
    raw = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    # The input is already a saved replay. Recreate lightweight records outside
    # the timing region, bypassing unrelated import validation and tensor copies.
    rows = []
    for episode in raw['training_state']['replay']['online']['episodes']:
        for state in episode['transitions']:
            row = object.__new__(MacroTransition)
            for key, value in state.items():
                if key == 'action_events' and value is not None:
                    value = tuple(ActionEvent(**event) for event in value)
                object.__setattr__(row, key, value)
            rows.append(row)
    variants = {
        'snapshot_reference': lambda: [reference_snapshot(row) for row in rows],
        'snapshot_optimized': lambda: [row.state_dict() for row in rows],
        'normalize_reference': lambda: reference_normalize(raw['training_state']),
        'normalize_optimized': lambda: _normalize_plain_value(raw['training_state'], path='training_state'),
    }
    timings = {key: [] for key in variants}
    for _ in range(args.repeats):
        for key, function in variants.items():
            start = time.perf_counter()
            result = function()
            timings[key].append(time.perf_counter() - start)
            del result
            print(key, timings[key][-1], flush=True)
    report = {
        'checkpoint': str(args.checkpoint.resolve()), 'device': 'cpu',
        'windows': len(rows), 'events': sum(len(row.action_events or ()) for row in rows),
        'seconds': timings,
        'median_seconds': {key: statistics.median(value) for key, value in timings.items()},
        'limitations': ['Preparation only; excludes serialization, fsync and GPU transfer.',
                        'Reference normalization omits diagnostic path formatting, conservatively favoring reference.'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
