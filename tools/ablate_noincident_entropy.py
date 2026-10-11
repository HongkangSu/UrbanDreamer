"""Controlled no-incident entropy ablation; never changes production state."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_entropy_ablation import evaluate, model_hash, tree_hash
from tools.run_online_pilot import atomic_json, sha256_file
from tools.train_codec_ready import SparseGraphCodec
from urban_dreamer.learning.training import MainlineTrainer, individual_actor_advantage, population_scaled_entropy, ACTOR_ENTROPY_COEF
from urban_dreamer.learning import parallel_imagination
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config


def gradient_probe(trainer, rows, arm, pass_index):
    """Probe the exact sampled actor graph before either critic or actor updates."""
    if not rows:
        return dict(pass_index=pass_index, events=0)
    advantage = individual_actor_advantage(
        torch.cat([row['global_advantage'] for row in rows]),
        torch.cat([row['timeout_return'] for row in rows])).detach()
    entropy = torch.cat([row['entropy'] for row in rows])
    population = torch.cat([row['entropy'].new_full(row['entropy'].shape, row['population']) for row in rows])
    bonuses, _ = population_scaled_entropy(entropy, population,
        torch.cat([row['global_advantage'] for row in rows]), base_coef=ACTOR_ENTROPY_COEF, mode=arm)
    policy_loss = -(torch.cat([row['log_prob'] for row in rows]) * advantage).mean()
    entropy_loss = -bonuses.mean()
    parameters = tuple(trainer.actor.parameters())
    policy_grad = torch.autograd.grad(policy_loss, parameters, retain_graph=True, allow_unused=True)
    entropy_grad = torch.autograd.grad(entropy_loss, parameters, retain_graph=True, allow_unused=True)
    def flat(values):
        return torch.cat([(torch.zeros_like(p) if value is None else value).reshape(-1)
                          for p, value in zip(parameters, values)]).detach()
    p, e = flat(policy_grad), flat(entropy_grad)
    pn, en = p.norm(), e.norm()
    cosine = float(torch.dot(p, e) / (pn * en)) if float(pn * en) > 0 else None
    return dict(pass_index=pass_index, events=len(advantage), return_policy_loss=float(policy_loss.detach()),
        entropy_loss=float(entropy_loss.detach()), return_gradient_norm=float(pn), entropy_gradient_norm=float(en),
        entropy_to_return_norm=float(en / pn) if float(pn) else None, gradient_cosine=cosine,
        population_min=float(population.min()), population_max=float(population.max()),
        actor_advantage_mean=float(advantage.mean()), actor_advantage_std=float(advantage.std(unbiased=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--passes', type=int, default=32)
    parser.add_argument('--gradient-only', action='store_true', help='One shared sampled graph, no updates or real evaluation')
    parser.add_argument('--arms', nargs='+', default=['legacy', 'population', 'off'])
    parser.add_argument('--source', type=Path, default=ROOT / 'outputs/noincident_decline_diagnosis_20261008/frozen')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/noincident_decline_diagnosis_20261008/entropy')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['URB_IMAGINATION_WORKERS'] = '8'
    torch.set_num_threads(4)
    torch.cuda.set_device(1)
    torch.cuda.set_per_process_memory_fraction(.08, 1)
    path = args.source / 'checkpoints/latest.pt'
    config = load_config(args.source / 'run_config.toml')
    if config.method.sp_prior_weight != 0:
        raise ValueError('experiment requires no prior')
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    if args.gradient_only:
        directory = args.output / 'matched_graph'
        directory.mkdir(parents=True, exist_ok=True)
        for name in ('offline_manifest.json', 'codec_validation_manifest.json'):
            shutil.copy2(args.source / name, directory / name)
        trainer = MainlineTrainer(config=replace(config, run=replace(config.run, output_dir=directory, device='cuda:1')),
            prepared_inputs=prepared, topology=topology)
        trainer.world_model.flow_projector.__class__ = SparseGraphCodec
        trainer.world_model.flow_projector.register_buffer('sparse_adjacency', trainer._tensor(topology.network.adjacency).to_sparse().coalesce(), persistent=False)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        trainer.load_state_dict(checkpoint['training_state'])
        del checkpoint
        trainer.world_model.requires_grad_(False)
        before = {name: model_hash(getattr(trainer, name)) for name in ('world_model', 'actor', 'latent_critic', 'event_critic')}
        try:
            trainer.world_model.eval()
            trainer.world_model.set_online_chunk_profile()
            event_rows, _ = parallel_imagination.collect_parallel_imagination(trainer, workers=8)
            probes = {arm: gradient_probe(trainer, event_rows, arm, 1) for arm in ('legacy', 'population', 'off')}
            after = {name: model_hash(getattr(trainer, name)) for name in before}
            if before != after:
                raise RuntimeError('matched graph probe changed weights')
            atomic_json(directory / 'result.json', dict(source_sha256=sha256_file(path), exactly_same_event_graph=True, updates=0, unchanged_model_hashes=before, probes=probes))
            print(json.dumps(probes), flush=True)
        finally:
            trainer.close()
        return
    source = torch.load(path, map_location='cpu', weights_only=False)
    protocol = dict(source=str(path), source_sha256=sha256_file(path), source_online_episodes=source['next_unit'],
        passes=args.passes, arms=args.arms, rollouts_per_pass=8, max_windows=8, world_model_updates=0,
        stochastic_evaluation_indices=[6, 7], demand_seed=314159, actor_sampling_seed=271828,
        experimental_override=True, note='Same checkpoint and initial RNG per arm. Real SUMO labels used only for evaluation; no labels fed to imagination. World model frozen. Critic and actor Adam states retained.')
    del source
    atomic_json(args.output / 'protocol.json', protocol)
    first_hashes = None
    for arm in args.arms:
        if arm not in {'legacy', 'population', 'off'}:
            raise ValueError('unknown entropy arm')
        os.environ['URB_ACTOR_ENTROPY_MODE'] = arm
        directory = args.output / arm
        directory.mkdir(parents=True, exist_ok=True)
        for name in ('offline_manifest.json', 'codec_validation_manifest.json'):
            shutil.copy2(args.source / name, directory / name)
        trainer = MainlineTrainer(config=replace(config, run=replace(config.run, output_dir=directory, device='cuda:1')),
            prepared_inputs=prepared, topology=topology)
        trainer.world_model.flow_projector.__class__ = SparseGraphCodec
        trainer.world_model.flow_projector.register_buffer('sparse_adjacency', trainer._tensor(topology.network.adjacency).to_sparse().coalesce(), persistent=False)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        trainer.load_state_dict(checkpoint['training_state'])
        del checkpoint
        trainer.world_model.requires_grad_(False)
        initial_state = trainer.state_dict()
        hashes = {key: tree_hash(initial_state[key]) for key in ('models', 'optimizers', 'rng', 'replay')}
        del initial_state
        if first_hashes is None:
            first_hashes = hashes
        if hashes != first_hashes:
            raise RuntimeError('initial controls differ between arms')
        atomic_json(directory / 'initial_control.json', hashes)
        initial_wm = model_hash(trainer.world_model)
        rows, probes = [], []
        original_collect = parallel_imagination.collect_parallel_imagination
        current_pass = 0
        def collect_with_probe(*positional, **kwargs):
            event_rows, losses = original_collect(*positional, **kwargs)
            if current_pass in {1, 16, args.passes}:
                probe = gradient_probe(trainer, event_rows, arm, current_pass)
                probes.append(probe)
                atomic_json(directory / 'gradient_probes.json', probes)
                print(json.dumps(dict(arm=arm, gradient_probe=probe)), flush=True)
            return event_rows, losses
        parallel_imagination.collect_parallel_imagination = collect_with_probe
        try:
            before = evaluate(trainer, args.output / 'before', indices=(6, 7))
            for current_pass in range(1, args.passes + 1):
                started = time.monotonic()
                metrics = trainer._train_imagination_pass()
                row = dict(pass_index=current_pass, metrics=metrics, wall_seconds=time.monotonic()-started)
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise RuntimeError('nonfinite training metrics')
                coefficient = metrics['entropy_coefficient_mean']
                if arm == 'legacy' and not math.isclose(coefficient, .002, rel_tol=1e-4):
                    raise RuntimeError('legacy coefficient mismatch')
                if arm == 'population' and not 0 < coefficient < .0001:
                    raise RuntimeError('population coefficient mismatch')
                if arm == 'off' and coefficient != 0:
                    raise RuntimeError('off coefficient mismatch')
                rows.append(row)
                with (directory / 'passes.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                print(json.dumps(dict(arm=arm, **row)), flush=True)
            final_wm = model_hash(trainer.world_model)
            if initial_wm != final_wm:
                raise RuntimeError('world model changed')
            torch.save(dict(arm=arm, passes_completed=args.passes, experimental_protocol=protocol,
                training_state=trainer.state_dict()), directory / 'experimental_state.pt')
            after = evaluate(trainer, directory / 'after', indices=(6, 7))
            if [(r['index'], r['demand_hash'], r['population']) for r in before] != [(r['index'], r['demand_hash'], r['population']) for r in after]:
                raise RuntimeError('paired evaluation demand differs')
            atomic_json(directory / 'result.json', dict(arm=arm, passes=args.passes, initial_hashes=hashes,
                world_model_hash=final_wm, world_model_unchanged=True, gradient_probes=probes, training=rows, before=before, after=after))
            print(json.dumps(dict(arm=arm, status='complete', after=after)), flush=True)
        finally:
            parallel_imagination.collect_parallel_imagination = original_collect
            trainer.close()
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
    atomic_json(args.output / 'status.json', dict(status='complete', arms=args.arms, passes=args.passes))


if __name__ == '__main__':
    main()
