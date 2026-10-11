"""Isolated boundary-only TraCI edge-read experiment; never mutates production sessions."""
from __future__ import annotations
import argparse
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot-dir', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--baseline-archive', type=Path, required=True)
    p.add_argument('--strategies', nargs='+', choices=('direct', 'transient'), default=['direct', 'transient'])
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(args.source_dir.resolve()))
    import numpy as np
    import torch
    from urban_dreamer.learning.training import MainlineTrainer
    from urban_dreamer.traffic.event_session import SumoEventSession, _EDGE_STATE_VARS
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.traffic.topology import NextHopCandidates, load_traffic_topology
    from urban_dreamer.workflow.checkpoint import load_checkpoint
    from urban_dreamer.workflow.config import load_config
    from urban_dreamer.workflow.pipeline import combined_input_hash

    class BoundarySession(SumoEventSession):
        strategy = 'direct'
        def _flow(self):
            if self._flow_cache is not None and self._flow_cache[0] == self._time_ms:
                return self._flow_cache[1]
            if not getattr(self, '_edge_unsubscribed', False):
                for edge in self.network.edge_ids:
                    self._connection.edge.unsubscribe(edge)
                self._edge_unsubscribed = True
            if self.strategy == 'direct':
                # Reuse the exact four-getter implementation, retaining TraCI connection.
                self.backend = 'libsumo'
                try:
                    return super()._flow()
                finally:
                    self.backend = 'traci'
            if not self.done:
                now = self._time_ms / 1000.
                for edge in self.network.edge_ids:
                    # Subscribe supplies current values synchronously; expiry avoids
                    # recomputing/transmitting this road's values on the next step.
                    self._connection.edge.subscribe(edge, _EDGE_STATE_VARS, begin=now, end=now)
            return super()._flow()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    config = load_config(args.snapshot_dir / 'run_config.toml')
    config = replace(config, run=replace(config.run, device='cpu', output_dir=output))
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    ckpt = load_checkpoint(output_dir=args.snapshot_dir,
        expected_config_fingerprint=config.semantic_fingerprint,
        expected_input_hash=combined_input_hash(prepared))
    trainer = MainlineTrainer(config=config, prepared_inputs=prepared, topology=topology,
                              _session_factory=BoundarySession)
    trainer.load_state_dict(ckpt.training_state)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), copy.deepcopy(trainer.rng.bit_generator.state))
    initial_metrics = copy.deepcopy(trainer.metrics)
    actor_hash = lambda: hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in trainer.actor.parameters())).hexdigest()
    protocol = dict(checkpoint=str(ckpt.path), checkpoint_sha256=hashlib.sha256(ckpt.path.read_bytes()).hexdigest(),
        source_dir=str(args.source_dir), actor_sha256=actor_hash(), demand_seed=42, demand_index=42,
        backend='traci', device='cpu', threads=2, baseline_archive=str(args.baseline_archive))
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows = []
    try:
        for strategy in args.strategies:
            BoundarySession.strategy = strategy
            folder = output / strategy
            trainer.config = replace(config, run=replace(config.run, output_dir=folder))
            trainer._next_environment_episode = 42
            trainer.candidates = NextHopCandidates(topology.network)
            trainer.metrics = copy.deepcopy(initial_metrics)
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
            trainer.rng.bit_generator.state = copy.deepcopy(rng[3])
            start = time.perf_counter()
            collected = trainer._run_episode(mode='online', episode_index=42)
            archive = folder/'episodes/online/episode_000042.npz'
            row = dict(strategy=strategy, wrapper_seconds=time.perf_counter()-start,
                metrics=collected.metrics, archive=str(archive), actor_unchanged=actor_hash()==protocol['actor_sha256'])
            if args.baseline_archive.exists():
                with np.load(args.baseline_archive, allow_pickle=False) as a, np.load(archive, allow_pickle=False) as b:
                    row['arrays_exact'] = {k:bool(np.array_equal(a[k], b[k])) for k in a.files if k != 'summary_json'}
                    sa=json.loads(a['summary_json'].tobytes()); sb=json.loads(b['summary_json'].tobytes())
                    row['travel_times_exact']=sa['travel_times']==sb['travel_times']
                    row['metrics_exact']={k:sa['metrics'][k]==sb['metrics'][k] for k in sa['metrics'] if not k.endswith('wall_seconds')}
                    row['demand_exact']=sa['metadata']['demand_sha256']==sb['metadata']['demand_sha256']
            rows.append(row)
            print(json.dumps(row), flush=True)
            (output/'results.json').write_text(json.dumps(dict(protocol=protocol,runs=rows),indent=2))
    finally:
        trainer.close()

if __name__ == '__main__':
    main()
