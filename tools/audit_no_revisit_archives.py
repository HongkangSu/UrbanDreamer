#!/usr/bin/env python3
"""Audit complete physical histories and observed decisions in no-revisit archives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np


def read_json(archive, member):
    with archive.open(member) as stream:
        return json.loads(np.lib.format.read_array(stream, allow_pickle=False).tobytes())


def audit(path):
    with zipfile.ZipFile(path) as archive:
        summary = read_json(archive, 'summary_json.npy')
        states = read_json(archive, 'states_json.npy')
        decisions = read_json(archive, 'decisions_json.npy')
    metadata = summary['metadata']
    if metadata.get('forbid_revisit') is not True:
        raise ValueError(f'archive is not in no-revisit mode: {path}')
    histories = {}
    for state in states:
        for vehicle in state['vehicles']:
            identity = str(vehicle['vehicle_id'])
            history = vehicle['visited_edges']
            previous = histories.get(identity, [])
            if history[:len(previous)] != previous or len(history) != len(set(history)):
                raise ValueError(f'physical history repeats or loses its prefix: {path}, {identity}')
            if vehicle['departed'] and (not history or history[-1] != vehicle['current_edge']):
                raise ValueError(f'physical edge/history mismatch: {path}, {identity}')
            histories[identity] = history
    seen = {}
    for decision in decisions:
        identity = str(decision['vehicle_id'])
        current, target = decision['current_edge'], decision['next_edge']
        history = histories[identity]
        position = history.index(current)
        if target in history[:position + 1] or current in seen.setdefault(identity, set()):
            raise ValueError(f'decision revisits a road: {path}, {identity}')
        seen[identity].add(current)
    if metadata.get('sp_prior_weight', 0.) != 0.:
        raise ValueError(f'policy prior enabled: {path}')
    return dict(path=str(path), city=metadata['city'], method=metadata['method'],
        episode=metadata['episode'], population=metadata['total_vehicle_count'],
        physical_edge_visits=sum(map(len, histories.values())), physical_revisits=0,
        decision_events=len(decisions), decision_revisits=0,
        completion_rate=summary['metrics']['completion_rate'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    paths = sorted({p for path in args.paths for p in (path.rglob('episode_*.npz') if path.is_dir() else [path])})
    results = [audit(path) for path in paths]
    text = json.dumps(dict(archives=len(results), results=results), indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
