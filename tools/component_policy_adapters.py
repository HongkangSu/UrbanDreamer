"""Frozen actor-only adapters using production selectors and audited snapshots.

Snapshot CLI copies a single open checkpoint inode; training may atomically replace
the pathname without changing the copied file. No training objects are constructed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from urban_dreamer.models.policy import NextHopActor, set_shortest_path_prior
from urban_dreamer.traffic.simulation_settings import physics_identity
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.pipeline import combined_input_hash


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def simulation_contract(config):
    """Physical dynamics and observation protocol; excludes learner parameters."""
    simulation = config.simulation
    fields = ('vehicle_max_speed_mps', 'incident_edge', 'incident_start_seconds',
              'incident_end_seconds', 'incident_speed_mps', 'demand_duration_seconds',
              'session_protocol', 'observation_step_seconds', 'backend')
    return dict(physics_identity(simulation), **{key: getattr(simulation, key) for key in fields},
                sumo_extra_args=list(simulation.sumo_extra_args))


def snapshot_checkpoint(*, source, config, output):
    """Freeze source config, full checkpoint bytes, provenance, and actor artifact."""
    source, config, output = Path(source).resolve(), Path(config).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / 'checkpoints/latest.pt'
    checkpoint.parent.mkdir()
    digest = hashlib.sha256()
    with source.open('rb') as reader, checkpoint.open('xb') as writer:
        before = os.fstat(reader.fileno())
        for block in iter(lambda: reader.read(1024 * 1024), b''):
            digest.update(block)
            writer.write(block)
        after = os.fstat(reader.fileno())
        writer.flush()
        os.fsync(writer.fileno())
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Checkpoint inode changed during copy')
    if sha256_file(checkpoint) != digest.hexdigest():
        raise ValueError('Checkpoint copy failed digest verification')
    # Loading from the original path preserves any relative network paths.
    source_config = load_config(config)
    prepared = validate_prepared_inputs(source_config)
    config_bytes = config.read_bytes()
    (output / 'source_run_config.toml').write_bytes(config_bytes)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('config_fingerprint') != source_config.semantic_fingerprint:
        raise ValueError('Snapshot checkpoint and source config fingerprints disagree')
    if 'training_state' in payload:
        from urban_dreamer.workflow.checkpoint import load_checkpoint
        checked = load_checkpoint(output_dir=output,
            expected_config_fingerprint=source_config.semantic_fingerprint,
            expected_input_hash=combined_input_hash(prepared))
        method = 'urban_dreamer'
        state = checked.training_state
        if state['reward_contract'] != 'causal_cumulative_mean_global_v1':
            raise ValueError('Urban checkpoint uses a different reward contract')
        actor = state['models']['actor']
        committed = int(state['metrics']['online_episodes_completed'])
        if checked.next_unit != committed:
            raise ValueError('Checkpoint next_unit differs from committed online episode count')
        episode = dict(committed_online_episodes=committed, next_unit=checked.next_unit,
                       next_environment_episode=state['next_environment_episode'], phase=checked.phase.value)
    else:
        from urban_dreamer.learning.baselines.common import load_checkpoint
        method = payload['method']
        if method not in ('mappo', 'ippo', 'dreamer'):
            raise ValueError(f'Unsupported actor-only baseline {method}')
        payload = load_checkpoint(checkpoint, method_name=method, prepared=prepared, config=source_config)
        if payload['args'].get('reward_objective') != 'observed_mean' or payload['args'].get('forbid_revisit', False):
            raise ValueError('Baseline checkpoint uses a different reward or action-mask contract')
        actor = payload['method_state']['actor']
        episode = dict(committed_online_episodes=int(payload['episode']), source_episode=int(payload['episode']))
    actor_path = output / 'actor.pt'
    torch.save(actor, actor_path)
    provenance = dict(schema_version=1, method=method, source_checkpoint=str(source),
        source_config=str(config), checkpoint_sha256=digest.hexdigest(), copied_inode=before.st_ino,
        copied_device=before.st_dev, copied_size=before.st_size, copied_mtime_ns=before.st_mtime_ns,
        source_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        actor_sha256=sha256_file(actor_path), input_hashes=prepared.input_hashes,
        combined_input_hash=combined_input_hash(prepared), config_fingerprint=source_config.semantic_fingerprint,
        simulation=simulation_contract(source_config), episode=episode,
        source_simulation=asdict(source_config.simulation),
        role='early development diagnostic; not a final-performance checkpoint',
        actor_recurrent_state=False, real_policy='actor-only; zero shortest-path prior; deterministic argmax',
        future_demand_access=False)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return provenance


def validate_contract(provenance, runner):
    if provenance['input_hashes'] != runner.prepared.input_hashes:
        raise ValueError('Policy snapshot network/input hashes differ from evaluation')
    if provenance['combined_input_hash'] != combined_input_hash(runner.prepared):
        raise ValueError('Policy snapshot combined input hash differs from evaluation')
    if provenance['simulation'] != simulation_contract(runner.config):
        raise ValueError('Policy snapshot physical/observation contract differs from evaluation')
    if getattr(runner.args, 'forbid_revisit', False):
        raise ValueError('These actor snapshots require the production revisit-allowed action space')


def actor_digest(actor):
    digest = hashlib.sha256()
    for key, tensor in sorted(actor.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenActorPolicy:
    def __init__(self, actor, method, audit):
        self.actor = actor.cpu().eval().requires_grad_(False)
        set_shortest_path_prior(self.actor, 0., 1.)
        self.device = torch.device('cpu')
        self.method = method
        self.audit = audit
        self.evaluation_temperature = None
        self._digest = actor_digest(actor)
        self._episode_seed = None
        self._selections = 0

    def begin_episode(self, *, seed):
        # Production selectors are feed-forward for these four methods, including
        # Dreamer: RSSM history is used in learning, not real-environment select.
        self.actor.eval()
        self._episode_seed = int(seed)
        self._selections = 0
        self._assert_frozen()

    def _assert_frozen(self):
        if self.actor.training or any(p.requires_grad for p in self.actor.parameters()):
            raise RuntimeError('Evaluation actor is not frozen/eval')
        if self.actor.prior_weight != 0 or actor_digest(self.actor) != self._digest:
            raise RuntimeError('Evaluation actor weights or shortest-path prior changed')

    def end_episode(self):
        self._assert_frozen()
        return dict(seed=self._episode_seed, select_calls=self._selections,
                    actor_state_sha256=self._digest, optimizer_updates=0, hidden_state='none; production feed-forward actor')

    def select(self, vehicles, actor_obs, frozen_flow, now, *, explore=False):
        if explore or self._episode_seed is None:
            raise ValueError('Frozen policy requires begin_episode and explore=False')
        self._selections += 1
        with torch.inference_mode():
            if self.method in ('mappo', 'ippo'):
                from urban_dreamer.learning.baselines.ppo import RoutingPpoMethod
                return RoutingPpoMethod.select(self, vehicles, actor_obs, frozen_flow, now, explore=False)
            if self.method == 'dreamer':
                from urban_dreamer.learning.baselines.dreamer import DreamerMethod
                return DreamerMethod.select(self, vehicles, actor_obs, frozen_flow, now, explore=False)
            if self.method == 'urban_dreamer':
                from urban_dreamer.learning.training import MainlineTrainer
                decisions = MainlineTrainer._select(self, vehicles, frozen_flow, now,
                    context=None, mode='evaluation', observations=actor_obs)
                choices = []
                for row, decision in enumerate(decisions):
                    indices = np.flatnonzero(actor_obs.mask[row] & (actor_obs.edge_indices[row] == decision.next_edge))
                    if len(indices) != 1:
                        raise ValueError('Production actor decision not uniquely present in legal candidates')
                    choices.append(int(indices[0]))
                return choices, [dict(entropy=0., evaluation_policy='zero_prior_argmax') for _ in choices]
        raise ValueError(f'Unsupported policy {self.method}')


def load_policy(*, checkpoint, runner):
    checkpoint = Path(checkpoint).resolve()
    root = checkpoint.parent.parent if checkpoint.name == 'latest.pt' else checkpoint.parent
    provenance = json.loads((root / 'provenance.json').read_text())
    full_checkpoint = root / 'checkpoints/latest.pt'
    if sha256_file(full_checkpoint) != provenance['checkpoint_sha256']:
        raise ValueError('Immutable full checkpoint digest mismatch')
    if sha256_file(root / 'actor.pt') != provenance['actor_sha256']:
        raise ValueError('Immutable actor artifact digest mismatch')
    if sha256_file(root / 'source_run_config.toml') != provenance['source_config_sha256']:
        raise ValueError('Immutable source configuration digest mismatch')
    validate_contract(provenance, runner)
    actor = NextHopActor()
    actor.load_state_dict(torch.load(root / 'actor.pt', map_location='cpu', weights_only=True), strict=True)
    return FrozenActorPolicy(actor, provenance['method'], provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(snapshot_checkpoint(source=args.source, config=args.config, output=args.output), indent=2))


if __name__ == '__main__':
    main()
