"""Closed-loop internal WM/vehicle consistency; this does not measure SUMO accuracy."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.component_experiment_data import atomic_json, dataset, load_frozen, sha256
from tools.component_prediction_metrics import grouped_bootstrap
from tools.evaluate_component_world_models import select_traffic_conditions, stable_seed
from urban_dreamer.learning.action_events import pack_action_events
from urban_dreamer.learning.imagination import advance_imagined_window, clip_road_speed_fractions
from urban_dreamer.learning.observations import build_actor_observations, prepare_actor_window
from urban_dreamer.models.experimental_dynamics import apply_variant, variant_metadata
from urban_dreamer.traffic.environment import NextHopDecision
from urban_dreamer.traffic.topology import NextHopCandidates


def active_cohort(states):
    selected = deepcopy([state for state in states if state['departed'] and not state['arrived']])
    if len({str(state['vehicle_id']) for state in selected}) != len(selected):
        raise ValueError('initial cohort contains duplicate vehicle IDs')
    return selected


def external_counts(states, edges):
    counts = np.zeros(edges, dtype=np.int64)
    for state in states:
        if state['departed'] and not state['arrived'] and not state.get('in_junction', False):
            edge = int(state['current_edge'])
            if not 0 <= edge < edges:
                raise ValueError('active external vehicle edge outside registry')
            counts[edge] += 1
    return counts


class DynamicShortestController:
    """Use the real actor observation builder and its causal dynamic route costs."""
    def __init__(self, candidates, *, deadline, initial_cohort, vehicle_max_speed_mps):
        self.candidates, self.deadline = candidates, deadline
        self.initial_cohort = initial_cohort
        self.vehicle_max_speed_mps = vehicle_max_speed_mps
        self.window = None

    def prepare(self, decoded):
        self.window = prepare_actor_window(decoded[:, [2, 1]], self.candidates.network.edge_count)

    def decide(self, pending, frozen_flow, now, states):
        active = sum(state['departed'] and not state['arrived'] for state in states)
        observations = build_actor_observations(policy_edge_features=self.window,
            vehicle_states=pending, candidates=self.candidates,
            episode_progress=min(now / max(self.deadline, 1.), 1.),
            active_vehicle_fraction=active / max(self.initial_cohort, 1),
            prior_cost_mode='dynamic', prior_speed_floor_mps=.5,
            prior_speed_cap_mps=self.vehicle_max_speed_mps)
        costs = np.where(observations.mask, observations.prior_costs_seconds, np.inf)
        choices = costs.argmin(-1)
        return [NextHopDecision(str(state['vehicle_id']), int(state['current_edge']),
            int(observations.edge_indices[i, choices[i]]), now) for i, state in enumerate(pending)]


def consistency_record(decoded, states, *, capacities, speed_limits, initial_size, cumulative_seconds,
                       cumulative_arrivals, elapsed, horizon, timestamp, deadline, traffic_model,
                       vehicle_max_speed_mps, window_seconds=0., window_arrivals=0):
    decoded = np.asarray(decoded)
    if not np.isfinite(decoded).all():
        raise ValueError('nonfinite decoded traffic state')
    count = decoded[:, 2] * capacities
    explicit = external_counts(states, len(count))
    active = sum(state['departed'] and not state['arrived'] for state in states)
    junction = sum(state['departed'] and not state['arrived'] and state.get('in_junction', False) for state in states)
    if explicit.sum() + junction != active or active + cumulative_arrivals != initial_size:
        raise ValueError('explicit fixed-cohort vehicle conservation failed')
    clipped = clip_road_speed_fractions(decoded[:, 1], traffic_model)
    raw_speeds = decoded[:, 1] * speed_limits
    used_speeds = np.minimum(clipped * speed_limits, vehicle_max_speed_mps)
    occupied_union = (explicit > 0) | (count > .5)
    return dict(horizon=horizon, elapsed_seconds=elapsed, timestamp=timestamp,
        training_deadline_seconds=deadline,
        beyond_training_deadline=timestamp > deadline, post_terminal_stress=active == 0,
        initial_cohort=initial_size, explicit_active=active, explicit_external=int(explicit.sum()),
        explicit_in_junction=junction, decoded_total_count=float(count.sum()),
        total_mass_bias=float(count.sum() - explicit.sum()),
        total_mass_absolute_error=float(abs(count.sum() - explicit.sum())),
        count_mae_all=float(np.abs(count - explicit).mean()),
        count_mae_occupied_union=(float(np.abs(count[occupied_union] - explicit[occupied_union]).mean())
                                 if occupied_union.any() else None),
        decoded_count_over_initial_cohort=max(0., float(count.sum() - initial_size)),
        decoded_count_over_active=max(0., float(count.sum() - active)),
        cumulative_arrivals=cumulative_arrivals, window_arrivals=window_arrivals,
        cumulative_active_vehicle_seconds=cumulative_seconds, window_active_vehicle_seconds=window_seconds,
        mean_remaining_minutes=cumulative_seconds / (60 * max(initial_size, 1)),
        negative_decoded_count_fraction=float((count < 0).mean()),
        negative_decoded_speed_fraction=float((decoded[:, 1] < 0).mean()),
        speed_fraction_clipped_fraction=float((decoded[:, 1] != clipped).mean()),
        speed_motion_clipped_fraction=float((np.abs(raw_speeds - used_speeds) > 1e-8).mean()))


@torch.no_grad()
def diagnose_rollout(model, initial_flow, initial_states, *, network, start_time, population,
                     horizons, steps, device='cpu', seed=0, deadline=1800.,
                     traffic_model='micro', vehicle_max_speed_mps=13.89, controller=None):
    """Advance explicit vehicles and WM together, using only their current states.

    The fixed episode population only normalizes action events as in production.
    No undispatched records or future replay frames are passed to the controller.
    Diagnostic extensions after all arrivals are labelled post-terminal stress.
    """
    if population < 1 or not horizons or min(horizons) < 1:
        raise ValueError('positive population and horizons required')
    states = active_cohort(initial_states)
    initial_ids = {str(state['vehicle_id']) for state in states}
    if not states:
        raise ValueError('imagination diagnostic requires at least one active seed vehicle')
    torch.manual_seed(seed)
    candidates = NextHopCandidates(network, traffic_model=traffic_model)
    controller = controller or DynamicShortestController(candidates, deadline=deadline,
        initial_cohort=len(states), vehicle_max_speed_mps=vehicle_max_speed_mps)
    latent = model.encode(torch.as_tensor(initial_flow, device=device, dtype=torch.float32).unsqueeze(0))
    capacities = np.maximum(np.asarray(network.capacities), 1.)
    limits = np.asarray(network.speed_limits_mps)
    cumulative, arrivals, event_index = 0., 0, 0
    rows, vehicle_trace = [], []
    for h in range(max(horizons) + 1):
        decoded = model.decode(latent)[0].cpu().numpy()
        now = start_time + h * 30.
        if h == 0 or h in horizons:
            rows.append(consistency_record(decoded, states, capacities=capacities, speed_limits=limits,
                initial_size=len(initial_ids), cumulative_seconds=cumulative, cumulative_arrivals=arrivals,
                elapsed=h * 30., horizon=h, timestamp=now, deadline=deadline, traffic_model=traffic_model,
                vehicle_max_speed_mps=vehicle_max_speed_mps,
                window_seconds=0. if h == 0 else result.active_vehicle_seconds,
                window_arrivals=0 if h == 0 else len(result.arrivals)))
            vehicle_trace.append(dict(horizon=h, external_counts=external_counts(states, network.edge_count).tolist(),
                decoded_counts=(decoded[:, 2] * capacities).tolist()))
        if h == max(horizons):
            break
        controller.prepare(decoded)
        speeds = np.minimum(clip_road_speed_fractions(decoded[:, 1], traffic_model) * limits,
                            vehicle_max_speed_mps)
        result = advance_imagined_window(states, candidates=candidates, edge_speeds_mps=speeds,
            frozen_flow=decoded, start_time=now, total_vehicle_count=population,
            decide=lambda pending, flow, when: controller.decide(pending, flow, when, states),
            interval_seconds=30., observation_step_seconds=1., session_protocol='event',
            event_index_start=event_index, traffic_model=traffic_model,
            vehicle_max_speed_mps=vehicle_max_speed_mps, stop_when_empty=False)
        if any(str(state['vehicle_id']) not in initial_ids for state in states) or np.any(result.departure):
            raise ValueError('future births entered a fixed-cohort rollout')
        integrated = sum((part.end_time - part.start_time) * part.active_count for part in result.cost_segments)
        if not np.isclose(integrated, result.active_vehicle_seconds):
            raise ValueError('window active seconds disagree with physical cost segments')
        cumulative += result.active_vehicle_seconds
        arrivals += len(result.arrivals)
        event_index += len(result.action_events)
        sparse = pack_action_events([[result.action_events]], window_starts=[[now]]).to(device)
        latent = model.transition(latent,
            torch.as_tensor(result.action, device=device, dtype=torch.float32).unsqueeze(0),
            action_events=sparse, steps=steps, solver='euler')
    return rows, vehicle_trace


def pairing_key(row):
    """Archive-unique condition/sample identity, independent of model predictions."""
    coordinates = [row[key] for key in ('demand_group', 'episode', 'start', 'condition_seed', 'sample')]
    return json.dumps(coordinates, ensure_ascii=True, separators=(',', ':'))


def validate_condition_identity(item, entries):
    index, start = item['episode_index'], item['start']
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(entries):
        raise ValueError('condition episode index outside current manifest')
    entry = entries[index]
    expected = dict(source_path=entry['path'], episode=f'{entry["path"]}::episode={entry["episode"]}',
                    demand_group=entry['demand_group'])
    if any(item.get(key) != value for key, value in expected.items()):
        raise ValueError('condition identity differs from current manifest; refusing cross-campaign/index mismatch')
    if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < entry['windows']:
        raise ValueError('condition start outside episode window range')


def summarize_consistency(records, *, seed):
    """Paired fixed-horizon estimand, with model-dependent strata diagnostic only."""
    if not records:
        raise ValueError('cannot summarize empty imagination records')
    by_horizon = defaultdict(list)
    for row in records:
        if 'training_deadline_seconds' in row:
            expected = row['timestamp'] > row['training_deadline_seconds']
            if row['beyond_training_deadline'] != expected:
                raise ValueError('deadline stratum disagrees with observed start/horizon timing')
        by_horizon[row['horizon']].append(row)
    reference_keys = sorted(pairing_key(row) for row in by_horizon[min(by_horizon)])
    if len(set(reference_keys)) != len(reference_keys):
        raise ValueError('duplicate condition/sample identities')
    for rows in by_horizon.values():
        if sorted(pairing_key(row) for row in rows) != reference_keys:
            raise ValueError('every horizon must retain the same complete condition/sample population')
    metrics = ('count_mae_all', 'count_mae_occupied_union', 'total_mass_absolute_error',
        'decoded_count_over_initial_cohort', 'cumulative_active_vehicle_seconds', 'cumulative_arrivals',
        'speed_motion_clipped_fraction')

    def aggregate(rows, horizon, deadline_scope, role, **details):
        keys = sorted(pairing_key(row) for row in rows)
        return dict(horizon=horizon, deadline_scope=deadline_scope, role=role, states=len(rows),
            matching_condition_sample_keys=keys,
            matching_keys_sha256=hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
            post_terminal_states=sum(row['post_terminal_stress'] for row in rows),
            metrics={key: grouped_bootstrap(rows, key, seed=seed) for key in metrics}, **details)

    fixed, terminal = [], []
    for h, rows in sorted(by_horizon.items()):
        for after_deadline in (False, True):
            selected = [row for row in rows if row['beyond_training_deadline'] == after_deadline]
            if not selected:
                continue
            scope = 'after_deadline' if after_deadline else 'within_deadline'
            role = 'after_deadline_stress' if after_deadline else 'primary_paired_fixed_horizon'
            fixed.append(aggregate(selected, h, scope, role))
            for ended in (False, True):
                subset = [row for row in selected if row['post_terminal_stress'] == ended]
                if subset:
                    terminal.append(aggregate(subset, h, scope, 'model_dependent_terminal_diagnostic',
                                              post_terminal_stress=ended))
    return dict(schema_version=2, status='complete', fixed_horizon=fixed, terminal_diagnostics=terminal,
        complete_condition_sample_keys=reference_keys,
        aggregation_protocol=dict(
            pairing_fields=['demand_group', 'episode', 'start', 'condition_seed', 'sample'],
            primary='within_deadline fixed horizon retaining all condition/sample keys, irrespective of terminal state',
            deadline='timestamp > training deadline; depends on initial timestamp/horizon, not model behavior',
            warning='fixed-horizon estimates include post-terminal stress and differ from production stopping; internal consistency is not SUMO accuracy',
            terminal_diagnostics='model-dependent membership; not a paired cross-model performance comparison',
            missing_horizons='rejected, rather than silently averaging a model-dependent subset'))


def resummarize(source_dir, output_dir):
    """Recompute only aggregates, preserving historical prediction artifacts."""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if source_dir.resolve() == output_dir.resolve() or output_dir.exists():
        raise ValueError('summary revision requires a new output directory')
    source_protocol = json.loads((source_dir / 'protocol.json').read_text())
    records = [json.loads(line) for line in (source_dir / 'states.jsonl').read_text().splitlines()]
    result = summarize_consistency(records, seed=source_protocol['campaign']['selection_seed'])
    revision = dict(kind='summary_only_revision_no_model_rerun', revised_at=datetime.now(timezone.utc).isoformat(),
        source_directory=str(source_dir.resolve()), code_sha256=sha256(__file__),
        source_states_sha256=sha256(source_dir / 'states.jsonl'),
        source_protocol_sha256=sha256(source_dir / 'protocol.json'),
        original_summary_sha256=sha256(source_dir / 'summary.json') if (source_dir / 'summary.json').exists() else None)
    output_dir.mkdir(parents=True)
    atomic_json(output_dir / 'revision.json', revision)
    atomic_json(output_dir / 'summary.json', dict(result, revision=revision))
    return result


def run(args):
    if args.resummarize_dir is not None:
        return resummarize(args.resummarize_dir, args.output_dir)
    if any((args.output_dir / name).exists() for name in ('protocol.json', 'states.jsonl')):
        raise ValueError('preserve prior diagnostic results; choose a new output directory')
    if args.samples < 1 or args.steps < 1:
        raise ValueError('samples and steps must be positive')
    torch.set_num_threads(args.threads)
    horizons = sorted(set(int(value) for value in args.horizons.split(',')))
    checkpoint = None if args.checkpoint is None else torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config, prepared, topology, model, protocol = load_frozen(args.experiment_dir, device='cpu',
        fresh_dynamics_seed=None if checkpoint is None else checkpoint['seed'])
    if checkpoint is not None:
        source = checkpoint['protocol'].get('parent', checkpoint['protocol'])
        if any(source[key] != protocol[key] for key in ('development_manifest_identity', 'checkpoint_sha256', 'config_sha256')):
            raise ValueError('component checkpoint protocol differs')
        apply_variant(model, checkpoint['variant'], checkpoint['seed'])
        model.load_state_dict(checkpoint['model'], strict=True)
    model.to(args.device).eval()
    data = dataset(args.experiment_dir, config, prepared, topology, split='validation', horizon=1)
    if args.conditions is None:
        conditions = select_traffic_conditions(data, torch.as_tensor(topology.network.capacities),
            per_group=args.conditions_per_group, max_groups=args.max_groups, horizon=1,
            seed=protocol['selection_seed'], include_empty=False)
    else:
        conditions = json.loads(args.conditions.read_text())
        conditions = [item for item in conditions if item.get('traffic_phase') != 'empty_stress']
        condition_protocol_path = args.conditions.parent / 'protocol.json'
        if condition_protocol_path.exists():
            condition_protocol = json.loads(condition_protocol_path.read_text())
            campaign = condition_protocol.get('campaign', condition_protocol)
            identity = campaign.get('development_manifest_identity')
            if identity is not None and identity != protocol['development_manifest_identity']:
                raise ValueError('conditions belong to a different campaign manifest')
    for item in conditions:
        validate_condition_identity(item, data.manifest['entries'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / 'conditions.json', conditions)
    atomic_json(args.output_dir / 'protocol.json', dict(schema_version=2, role='development_internal_consistency',
        code_sha256=sha256(__file__), summary_schema_version=2,
        campaign=protocol, model=variant_metadata(model), policy='dynamic_shortest_observed_decoded_speeds',
        checkpoint_sha256=protocol['checkpoint_sha256'] if checkpoint is None else sha256(args.checkpoint),
        samples=args.samples, steps=args.steps, solver='euler', horizons=horizons,
        action_population='fixed recorded episode population, matching training normalization',
        vehicles='only initial departed nonarrived cohort; count excludes internal junction links',
        observation='production build_actor_observations; no real future states or future births',
        terminal='continue after empty solely as labelled post_terminal_stress',
        conclusion_limit='internal decoded/explicit agreement does not establish real SUMO forecast or routing accuracy'))
    records = []
    with (args.output_dir / 'states.jsonl').open('w') as stream, (args.output_dir / 'road_counts.jsonl').open('w') as roads:
        for index, item in enumerate(conditions):
            batch = data.batch_from_starts(item['episode_index'], [item['start']], fm_only=False)
            for sample in range(args.samples):
                rows, traces = diagnose_rollout(model, batch.flow[0, 0], batch.vehicles[0],
                    network=topology.network, start_time=float(batch.time[0]), population=int(batch.total_vehicle_count[0]),
                    horizons=horizons, steps=args.steps, device=args.device,
                    seed=stable_seed(item['condition_seed'], sample), deadline=data.target_deadline_seconds,
                    traffic_model=config.simulation.traffic_model,
                    vehicle_max_speed_mps=config.simulation.vehicle_max_speed_mps)
                for row, trace in zip(rows, traces):
                    metadata = dict(item, sample=sample)
                    records.append(dict(metadata, **row))
                    stream.write(json.dumps(records[-1], allow_nan=False) + '\n')
                    roads.write(json.dumps(dict(metadata, **trace), allow_nan=False) + '\n')
                stream.flush()
                roads.flush()
            print(json.dumps(dict(conditions_done=index + 1, conditions_total=len(conditions))), flush=True)
    summary = summarize_consistency(records, seed=protocol['selection_seed'])
    atomic_json(args.output_dir / 'summary.json', dict(summary, conditions=len(conditions), code_sha256=sha256(__file__)))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--experiment-dir', type=Path, default=Path('outputs/component_experiments_20261011'))
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument('--resummarize-dir', type=Path,
                        help='read existing states and write a provenance-labelled summary revision only')
    result.add_argument('--checkpoint', type=Path)
    result.add_argument('--conditions', type=Path)
    result.add_argument('--conditions-per-group', type=int, default=3)
    result.add_argument('--max-groups', type=int, default=1)
    result.add_argument('--samples', type=int, default=1)
    result.add_argument('--horizons', default='1,2,4,8,16')
    result.add_argument('--steps', type=int, default=20)
    result.add_argument('--device', default='cpu')
    result.add_argument('--threads', type=int, default=2)
    return result


if __name__ == '__main__':
    run(parser().parse_args())
