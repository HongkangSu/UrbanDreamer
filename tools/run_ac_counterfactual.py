"""Frozen AC next-hop and bottleneck interventions with audited prefix replay.

Use the immutable source and actor from the existing temperature experiment.
No training, checkpoint migration, or state-file restoration is performed.
Every branch replays the entire identical prefix, including queued future demand.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/city_baselines_noincident_noprior_penalty_5000ep_20261008/source'
TRACKING = ROOT / 'outputs/ingo_temperature_decomposition_20261010'
ACTOR = ROOT / 'outputs/ingo_temperature_sweep_20261010/online_ac_actor.pt'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRACKING))
sys.path.insert(0, str(SOURCE))

import numpy as np
import torch
from path_tracking import RouteTrackingSession
from urban_dreamer.learning.baselines.common import EpisodeRunner
from urban_dreamer.models.policy import NextHopActor, set_shortest_path_prior
from urban_dreamer.traffic.simulation_settings import physics_identity


PREFIX_SEED = 81001
JOINT_START = 600.
JOINT_END = 1800.
FRACTIONS = (0., .05, .10, .20)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def keyed_uniform(seed, vehicle_id, decision_number):
    """One common variate per vehicle decision, independent of batch ordering."""
    payload = json.dumps([int(seed), str(vehicle_id), int(decision_number)], separators=(',', ':'))
    integer = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big') >> 12
    return (integer + .5) / 2**52


def sample_indices(probabilities, vehicles, counters, seed):
    indices = []
    for probability, vehicle in zip(probabilities, vehicles):
        vid = str(vehicle['vehicle_id'])
        number = counters[vid]
        counters[vid] += 1
        uniform = keyed_uniform(seed, vid, number)
        cumulative = np.cumsum(probability, dtype=np.float64)
        # Select among the positive mass only, including floating roundoff at 1.
        positive = np.flatnonzero(probability > 0)
        index = min(int(np.searchsorted(cumulative, uniform, side='right')), int(positive[-1]))
        indices.append(index)
    return indices


def ranked_subset(vehicle_ids, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError('fraction must be in [0, 1]')
    ordered = sorted(set(vehicle_ids), key=lambda v: (keyed_uniform(90210, v, 0), v))
    return set(ordered[:round(fraction * len(ordered))])


def initialize(temperature):
    settings = argparse.Namespace(config=SOURCE / 'configs/ingolstadt_custom.toml', device='cpu',
        threads=1, seed=42, vehicle_limit=0, session=None, backend=None, step_seconds=None,
        forbid_revisit=False, sp_prior_weight=0., global_reward_only=False)
    runner = EpisodeRunner(settings)
    state = torch.load(ACTOR, map_location='cpu', weights_only=False)
    assert state['method'] == 'online_ac' and state['episode'] == 5000
    assert state['input_hashes'] == runner.prepared.input_hashes
    assert state['config_fingerprint'] == runner.config.semantic_fingerprint
    assert state['physics'] == physics_identity(runner.config.simulation)
    assert runner.deadline == 3810 and runner.demand.duration_seconds == 1800
    actor = NextHopActor()
    actor.load_state_dict(state['actor'])
    actor.eval().requires_grad_(False)
    set_shortest_path_prior(actor, 0., 30.)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('this experiment requires a positive frozen policy temperature')
    return runner, actor


def alternate(runner, vehicle, chosen, legal, bottleneck):
    """Cheapest legal next hop whose planned suffix bypasses the fixed edge."""
    current, destination = int(vehicle['current_edge']), int(vehicle['destination_edge'])
    costs = runner.candidates.costs(destination)
    ff = runner.network.free_flow_travel_seconds
    limit = float(ff[chosen] + costs[chosen]) + 180.
    alternatives = sorted((edge for edge in legal if edge != chosen),
                          key=lambda e: (float(ff[e] + costs[e]), e))
    for edge in alternatives:
        if float(ff[edge] + costs[edge]) > limit:
            continue
        route = runner.candidates.full_route(current, edge, destination)
        if bottleneck not in route[1:]:
            return int(edge)
    return None


def snapshot(session, path, counters, decision_hash, frozen_flow):
    """Audit simulator RNG/queues/routes and the application's control state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    session._connection.simulation.saveState(str(path))
    with gzip.open(path, 'rb') as stream:
        tree = ET.fromstring(stream.read())
    # XML comments contain timestamps/options; actual simulation nodes do not.
    sumo_hash = hashlib.sha256(ET.tostring(tree)).hexdigest()
    payload = dict(time_ms=session._time_ms, vehicles=session._vehicles,
        active=sorted(session._active), pending=sorted(session._pending),
        travel_times=session.travel_times, actual_routes=session._actual_routes,
        route_history=session.route_history, audit_cursors=session.audit_cursors,
        frozen_flow=np.asarray(frozen_flow).tolist(), policy_counters=dict(counters),
        decision_prefix_sha256=decision_hash.hexdigest(), incident_active=session._incident_active,
        metrics={k: v for k, v in session.metrics.items() if k != 'wall_seconds'})
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    python_hash = hashlib.sha256(encoded).hexdigest()
    save_json(path.with_suffix('.audit.json'), payload)
    return dict(time=session._time_ms / 1000., sumo_sha256=sumo_hash, python_sha256=python_hash,
                decision_prefix_sha256=decision_hash.hexdigest(), state_file=str(path.relative_to(ROOT)))


def run_branch(runner, actor, *, output, seed, temperature, label, future_seed=PREFIX_SEED,
               target=None, action=None, bottleneck=None, target_ids=(), fraction=0., discover=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counters = Counter()
    trace = hashlib.sha256()
    probes, captures, events, interventions = [], {}, [], []
    eligible_seen, intervened = set(), set()
    congestion = np.zeros(runner.network.edge_count, np.float64)
    target_ids = set(target_ids)
    switch_time = float(target['time']) if target else JOINT_START
    single_snapshot_done = False

    class Probe(RouteTrackingSession):
        def advance_window(self, callback):
            flow = self._flow()
            if discover:
                count = np.asarray(flow[:, 2]) * self.network.capacities
                speed_ratio = np.asarray(flow[:, 1])
                congestion[:] += 30. * count * np.clip(1. - speed_ratio, 0., 1.)
            if not target and not discover and self._time_ms == int(JOINT_START * 1000):
                captures['state'] = snapshot(self, output / f'{label}_state.xml.gz', counters, trace, flow)
            return super().advance_window(callback)

    def factory(config, prepared, topology):
        simulation = replace(config.simulation, sumo_extra_args=tuple(config.simulation.sumo_extra_args) +
                             ('--save-state.rng', 'true', '--save-state.precision', '17'))
        probe = Probe(replace(config, simulation=simulation), prepared, topology,
                      step_seconds=1., backend='libsumo', forbid_revisit=False)
        probes.append(probe)
        return probe

    def select(vehicles, obs, frozen_flow, now):
        nonlocal single_snapshot_done
        session = probes[0]
        if target and now == target['time'] and not single_snapshot_done:
            assert any(v['vehicle_id'] == target['vehicle_id'] for v in vehicles)
            captures['state'] = snapshot(session, output / f'{label}_state.xml.gz', counters, trace, frozen_flow)
            single_snapshot_done = True
        # Every action at the target instant shares the prefix draw; only later
        # decisions use independently sampled continuation policy noise.
        policy_seed = future_seed if (now > switch_time if target else now >= switch_time) else PREFIX_SEED
        if discover:
            policy_seed = PREFIX_SEED
        with torch.inference_mode():
            logits = actor.forward_logits(torch.as_tensor(obs.features, dtype=torch.float32),
                                         torch.as_tensor(obs.mask))
            probabilities = (logits / temperature).softmax(-1).numpy()
        indices = sample_indices(probabilities, vehicles, counters, policy_seed)
        for i, (vehicle, index) in enumerate(zip(vehicles, indices)):
            current = int(vehicle['current_edge'])
            chosen = int(obs.edge_indices[i, index])
            legal = [int(e) for e, valid in zip(obs.edge_indices[i], obs.mask[i]) if valid]
            vid = str(vehicle['vehicle_id'])
            if discover and JOINT_START <= now < JOINT_END and len(legal) > 1:
                fixed = chosen if bottleneck is None else bottleneck
                if chosen == fixed:
                    alt = alternate(runner, vehicle, chosen, legal, fixed)
                    if alt is not None:
                        events.append(dict(time=float(now), vehicle_id=vid, current_edge=current,
                            destination_edge=int(vehicle['destination_edge']), a1=chosen, a2=alt,
                            legal=legal))
            if target and now == target['time'] and vid == target['vehicle_id']:
                assert current == target['current_edge'] and action in legal
                indices[i] = legal_index = list(obs.edge_indices[i]).index(action)
                assert obs.mask[i, legal_index]
                interventions.append(dict(time=float(now), vehicle_id=vid, before=chosen, after=action))
                chosen = action
            elif not target and not discover and JOINT_START <= now < JOINT_END and chosen == bottleneck:
                alt = alternate(runner, vehicle, chosen, legal, bottleneck)
                if alt is not None:
                    eligible_seen.add(vid)
                    if vid in target_ids and vid not in intervened:
                        indices[i] = list(obs.edge_indices[i]).index(alt)
                        interventions.append(dict(time=float(now), vehicle_id=vid, before=chosen, after=alt))
                        intervened.add(vid)
                        chosen = alt
            trace.update(json.dumps([float(now), vid, current, chosen], separators=(',', ':')).encode())
        return indices

    _, _, metrics = runner.run_episode(select, seed=seed, label=label, session_factory=factory)
    probe = probes[0]
    metrics = {k: v for k, v in metrics.items() if k not in ('choices', 'branch_choices')}
    probe.save_routes(output / f'{label}_routes.jsonl.gz')
    fields = ('travel_seconds', 'shortest_ff_seconds', 'actual_ff_seconds',
              'route_excess_seconds', 'residual_delay_seconds')
    metrics['decomposition'] = {key: float(np.mean([r[key] for r in probe.decomposition])) for key in fields}
    metrics['trip_time_complete'] = metrics['unfinished'] == 0
    if metrics['trip_time_complete']:
        assert abs(metrics['global_reward_sum'] + metrics['mean_travel_seconds'] / 60.) < 1e-9
    metrics.update(demand_seed=seed, future_seed=future_seed, label=label, fraction=fraction,
        state=captures.get('state'), intervention_count=len(interventions),
        assigned_target_count=len(target_ids), eligible_vehicle_count=len(eligible_seen),
        intervention_uptake_fraction=len(intervened) / len(target_ids) if target_ids else 0.,
        interventions=interventions,
        routes=f'{label}_routes.jsonl.gz')
    save_json(output / f'{label}.json', metrics)
    print(json.dumps(dict(seed=seed, label=label, mean_travel_seconds=metrics['mean_travel_seconds'],
        unfinished=metrics['unfinished'], intervention_count=len(interventions),
        seconds=round(metrics['collection_seconds'], 2))), flush=True)
    return metrics, events, congestion


def discover_seed(seed, output, temperature, bottleneck):
    runner, actor = initialize(temperature)
    metrics, events, burden = run_branch(runner, actor, output=output, seed=seed,
        temperature=temperature, label='discovery', bottleneck=bottleneck, discover=True)
    if bottleneck is None:
        counts = Counter(e['a1'] for e in events)
        viable = [edge for edge, count in counts.items() if count >= 20]
        if not viable:
            raise RuntimeError('no bottleneck with at least 20 legal bypass decision events')
        bottleneck = max(viable, key=lambda edge: (burden[edge], counts[edge], -edge))
        events = [e for e in events if e['a1'] == bottleneck]
    if not events:
        raise RuntimeError(f'no eligible bypass state in demand {seed}')
    target = next(e for e in events if e['time'] <= 1200.)
    info = dict(demand_seed=seed, demand_sha256=metrics['demand_sha256'],
        bottleneck=bottleneck, bottleneck_edge_id=runner.network.edge_ids[bottleneck],
        pilot_congestion_burden=float(burden[bottleneck]),
        eligible_event_count=len(events), target_ids=sorted({e['vehicle_id'] for e in events}),
        target=target, discovery=metrics)
    save_json(Path(output) / 'discovery_plan.json', info)
    return info


def evaluate_seed(seed, output, temperature, repeats):
    output = Path(output)
    info = json.loads((output / 'discovery_plan.json').read_text())
    runner, actor = initialize(temperature)
    pairs, joint = [], []
    expected_single = expected_joint = None
    for repeat in range(repeats):
        future_seed = PREFIX_SEED + repeat
        arms = []
        for name in ('a1', 'a2'):
            metrics, _, _ = run_branch(runner, actor, output=output, seed=seed,
                temperature=temperature, label=f'single_{repeat:02d}_{name}',
                future_seed=future_seed, target=info['target'], action=info['target'][name])
            assert metrics['demand_sha256'] == info['demand_sha256']
            identity = {k: v for k, v in metrics['state'].items() if k != 'state_file'}
            if expected_single is None:
                expected_single = identity
            assert identity == expected_single, (seed, name, identity, expected_single)
            assert metrics['intervention_count'] == 1
            if repeat == 0 and name == 'a1':
                for field in ('event_count', 'exposure_seconds', 'global_reward_sum', 'mean_travel_seconds'):
                    assert metrics[field] == info['discovery'][field], (seed, field)
            arms.append(metrics)
        a, b = arms
        pairs.append(dict(demand_seed=seed, repeat=repeat, future_seed=future_seed,
            difference=a['mean_travel_seconds'] - b['mean_travel_seconds'],
            reward_difference=b['global_reward_sum'] - a['global_reward_sum'],
            a1=a, a2=b))
        save_json(output / 'single_pairs.json', pairs)
    # Three stochastic continuations per demand, same S at 600s for all doses.
    for repeat in range(min(repeats, 3)):
        arms = []
        for fraction in FRACTIONS:
            selected = ranked_subset(info['target_ids'], fraction)
            metrics, _, _ = run_branch(runner, actor, output=output, seed=seed,
                temperature=temperature, label=f'joint_{repeat:02d}_{round(100*fraction):02d}',
                future_seed=PREFIX_SEED + repeat, bottleneck=info['bottleneck'],
                target_ids=selected, fraction=fraction)
            assert metrics['demand_sha256'] == info['demand_sha256']
            identity = {k: v for k, v in metrics['state'].items() if k != 'state_file'}
            if expected_joint is None:
                expected_joint = identity
            assert identity == expected_joint, (seed, fraction, identity, expected_joint)
            if repeat == 0 and fraction == 0:
                for field in ('event_count', 'exposure_seconds', 'global_reward_sum', 'mean_travel_seconds'):
                    assert metrics[field] == info['discovery'][field], (seed, field)
            arms.append(metrics)
        for treatment in arms[1:]:
            joint.append(dict(demand_seed=seed, repeat=repeat, fraction=treatment['fraction'],
                difference=arms[0]['mean_travel_seconds'] - treatment['mean_travel_seconds'],
                reward_difference=treatment['global_reward_sum'] - arms[0]['global_reward_sum'],
                baseline=arms[0], treatment=treatment))
        save_json(output / 'joint_pairs.json', joint)
    return dict(demand_seed=seed, single_pairs=len(pairs), joint_pairs=len(joint),
                prefix_states_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/ac_counterfactual_20261010')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(70042, 70052)))
    parser.add_argument('--temperature', type=float, default=.2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--discover-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.repeats < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('workers and repeats must be positive; demand seeds must be unique')
    output = args.output.resolve()
    if output.exists() and (output / 'protocol.json').exists():
        raise ValueError('use a fresh output directory; refusing to overwrite an experiment')
    output.mkdir(parents=True, exist_ok=True)
    runner, _ = initialize(args.temperature)
    protocol = dict(started_at=datetime.now(timezone.utc).isoformat(), method='online_ac',
        checkpoint_episode=5000, actor=str(ACTOR), actor_sha256=digest(ACTOR),
        source=str(SOURCE), source_session_sha256=digest(SOURCE / 'urban_dreamer/traffic/event_session.py'),
        config_fingerprint=runner.config.semantic_fingerprint, input_hashes=runner.prepared.input_hashes,
        physics=runner.simulation_settings, demand_seeds=args.seeds, future_policy_seeds=list(range(PREFIX_SEED, PREFIX_SEED+args.repeats)),
        temperature=args.temperature, training_updates=0, device='cpu', workers=args.workers,
        deadline_seconds=runner.deadline, demand_release_seconds=1800,
        joint_start=JOINT_START, joint_end=JOINT_END, fractions=FRACTIONS,
        baseline_prefix_policy_seed=PREFIX_SEED,
        random_coupling='SHA256(policy_seed, vehicle_id, decision_ordinal), inverse categorical CDF',
        state_recovery='Exact complete-demand prefix replay; RNG-inclusive SUMO and Python state hashes audited before intervention',
        future_demand='Identical realized full Poisson instance in every paired branch, including scheduled future vehicles',
        bottleneck_selection='Highest pilot 30s count*(1-normalized_speed) burden among edges with >=20 bypassable decision events, fixed before treatment runs',
        single_selection='First eligible pilot event >=600s and <=1200s at fixed bottleneck; a1=pilot sampled action, a2=cheapest legal bypass with <=180s added planned free-flow cost',
        joint_selection='Nested hashed exact subsets of pilot eligible vehicles, override once at first live eligible decision during [600,1800); later actor decisions unconstrained',
        inference='Conditional continuation noise within S; independent demand clusters for overall uncertainty; pilot seed included and also separately excluded',
        caveat='Policy distribution equals frozen T=0.2 AC, but keyed random stream differs from prior batch-multinomial temperature sweep. Planned bypass is a one-hop intervention, not a forced full trip route.')
    save_json(output / 'protocol.json', protocol)
    first = discover_seed(args.seeds[0], output / f'seed_{args.seeds[0]}', args.temperature, None)
    protocol.update(bottleneck=first['bottleneck'], bottleneck_edge_id=first['bottleneck_edge_id'])
    save_json(output / 'protocol.json', protocol)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(discover_seed, seed, output / f'seed_{seed}', args.temperature,
                               first['bottleneck']): seed for seed in args.seeds[1:]}
        for future in as_completed(futures):
            info = future.result()
            print(json.dumps(dict(stage='discovered', seed=info['demand_seed'],
                targets=len(info['target_ids']), time=info['target']['time'])), flush=True)
    if args.discover_only:
        return
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_seed, seed, output / f'seed_{seed}', args.temperature,
                               args.repeats): seed for seed in args.seeds}
        for future in as_completed(futures):
            completed.append(future.result())
            save_json(output / 'progress.json', dict(completed=completed, demand_count=len(args.seeds)))
    assert digest(ACTOR) == protocol['actor_sha256']
    save_json(output / 'completed.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        actor_unchanged=True, training_updates=0, results=completed))


if __name__ == '__main__':
    main()
