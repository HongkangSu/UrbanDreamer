"""CPU-only first-window attribution, without a latent dynamics transition.

The saved decoder output first reproduces the archived executor exactly. A
fresh CPU codec reconstruction and the observed seed flow then run the same
executor. Changing flow also changes the causal shortest-path controller, so
the contrast is a total initial-flow intervention, not a speed-only effect.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE = 'dd11390539935eeb87df34dfadea2187c714852fa806fb5f6bd8cbe2a598a739'
EXPECTED_CHECKPOINT = '2f63f0b2a5fe18f3a3bd9be27ffd1e63a2755464e7f2944e1227334f0ffe6af6'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def compare_events(left, right):
    """Align causal events by vehicle and occurrence, keeping unmatched events."""
    def indexed(rows):
        counter, result = {}, {}
        for row in rows:
            name = row['vehicle_id']
            ordinal = counter.get(name, 0)
            counter[name] = ordinal + 1
            result[(name, ordinal)] = row
        return result
    a, b = indexed(left), indexed(right)
    common = sorted(a.keys() & b.keys())
    return dict(left_events=len(left), right_events=len(right), matched_occurrences=len(common),
        next_hop_changes=sum(a[key]['next_edge'] != b[key]['next_edge'] for key in common),
        source_edge_changes=sum(a[key]['current_edge'] != b[key]['current_edge'] for key in common),
        timestamp_changes=sum(a[key]['decision_time'] != b[key]['decision_time'] for key in common),
        only_left=len(a.keys()-b.keys()), only_right=len(b.keys()-a.keys()),
        event_sequences_equal=left == right)


def first_window(trial, snapshot, topology, controls, flow, *, executor_seed, max_speed):
    """Replay the frozen operational executor, with one fresh private arbiter."""
    network = topology.network
    before_snapshot = trial.fidelity.fingerprint(snapshot.session)
    before_rng = trial.fidelity.fingerprint(trial.fidelity.external_rng_state())
    candidates = trial.fidelity.NextHopCandidates(network, traffic_model='meso')
    controller = trial.GuardedController(network, candidates, max_speed)
    states = [copy.deepcopy(row) for row in snapshot.session['_vehicles'].values()
              if row['departed'] and not row['arrived']]
    start = snapshot.session['_time_ms']/1000.
    arbiter = trial.fidelity.ChronologicalJunctionArbiter(controls, seed=executor_seed, uncertain_order=True)
    controller.prepare(flow, start)

    def decide(pending, current, now):
        rows = [{key: value for key, value in row.items() if not key.startswith('_city_')}
                for row in pending]
        return controller(rows, current, now)

    speed = np.minimum(trial.fidelity.clip_road_speed_fractions(flow[:, 1], 'meso') *
                       np.maximum(np.asarray(network.speed_limits_mps), .1), max_speed)
    physical = trial.fidelity.advance_city_imagined_window(states, candidates=candidates,
        edge_speeds_mps=speed, frozen_flow=flow, start_time=start,
        total_vehicle_count=len(snapshot.session['_vehicles']), decide=decide,
        junction_arbiter=arbiter, interval_seconds=30., observation_step_seconds=1.,
        session_protocol='event', traffic_model='meso', vehicle_max_speed_mps=max_speed,
        stop_when_empty=False)
    if (trial.fidelity.fingerprint(snapshot.session) != before_snapshot or
            trial.fidelity.fingerprint(trial.fidelity.external_rng_state()) != before_rng):
        raise RuntimeError('first-window attribution changed shared snapshot or external RNG')
    counts = np.bincount([int(row['current_edge']) for row in states
        if not row['arrived'] and not row.get('in_junction', False)], minlength=network.edge_count)
    exposure = {str(row['vehicle_id']): min(30., max(0., float(row['arrival_time'])-start))
                if row['arrived'] else 30. for row in states}
    if sum(exposure.values()) != physical.active_vehicle_seconds:
        raise ValueError('per-vehicle censored exposure does not reconcile with executor cost')
    return dict(vehicle_seconds=physical.active_vehicle_seconds, cohort_size=len(states),
        mean_minutes=physical.active_vehicle_seconds/len(states)/60.,
        per_vehicle_exposure_seconds=exposure, explicit_cohort_count=counts.tolist(),
        arrivals=sum(row['arrived'] for row in states),
        own_action_events=[asdict(event) for event in physical.action_events],
        executor_audit=arbiter.audit, effective_speeds_mps=speed.tolist(),
        controller_input_audit=controller.input_audit,
        snapshot_unchanged=True, external_rng_unchanged=True)


def assert_saved_reproduction(result, saved):
    expected = saved['windows'][0]
    if result['vehicle_seconds'] != expected['vehicle_seconds']:
        raise ValueError('saved first-window exposure was not reproduced exactly')
    if result['own_action_events'] != expected['own_action_events']:
        raise ValueError('saved first-window causal action events were not reproduced exactly')
    if result['explicit_cohort_count'] != saved['frames'][1]['explicit_cohort_count']:
        raise ValueError('saved first-window explicit endpoint counts were not reproduced exactly')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT/'outputs/component_experiments_20261011')
    parser.add_argument('--output', type=Path,
        default=ROOT/'outputs/pretrained_evaluation_20261012/reports/initial_executor_attribution_v1')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError('preserve existing report; choose a new output directory')
    source = args.campaign/'conditional_futures_source_v1'
    if sha(source/'manifest.json') != EXPECTED_SOURCE:
        raise ValueError('frozen conditional future source manifest differs')
    manifest = json.loads((source/'manifest.json').read_text())
    if not all(sha(source/name) == digest for name, digest in manifest['source_sha256'].items()):
        raise ValueError('frozen source bytes changed')
    sys.path.insert(0, str(source))
    from tools import evaluate_pretrained_conditional_futures as trial
    if not Path(trial.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError('wrong runtime imported instead of frozen conditional source')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    trial.fidelity.validate_pretrained(args.campaign)
    config, prepared, topology, model, _ = trial.load_frozen(args.campaign, device='cpu')
    model.eval().requires_grad_(False)
    controls = trial.fidelity.CityControlTable.from_network(prepared.network_xml,
        topology.network.edge_ids, no_external_overrides=True, verified_road_only_no_trains=False)
    if not all(row.support == 'exact_control' for row in controls.movements.values()):
        raise ValueError('unsupported city control')
    four = args.campaign/'conditional_futures_pretrained_fourstate_v1'
    plan = json.loads((four/'launch_plan.json').read_text())
    if [(row['seed'], row['time']) for row in plan['conditions']] != [
            (620001, 300), (620001, 900), (620002, 300), (620002, 900)]:
        raise ValueError('four-state evaluation identity differs')
    args.output.mkdir(parents=True)
    artifacts, records = {}, []
    for row in plan['conditions']:
        condition = f"seed{row['seed']}_t{row['time']}"
        directory = (args.campaign/'conditional_futures_pretrained_pilot_v1'
            if (row['seed'], row['time']) == (620001, 300) else four/'runs'/condition)
        snapshot_path = directory/'real/initial_python.pt'
        snapshot = trial.fidelity.BranchSnapshot(**torch.load(snapshot_path, map_location='cpu', weights_only=False))
        protocol = json.loads((directory/'protocol.json').read_text())
        if protocol['checkpoint_sha256'] != EXPECTED_CHECKPOINT or protocol['executor_seed'] != 920001:
            raise ValueError('saved condition checkpoint or executor seed differs')
        saved_models = [json.loads(path.read_text()) for path in sorted((directory/'model').glob('*.json'))]
        if len(saved_models) != 33:
            raise ValueError('expected all 32 sampled trajectories and zero-noise reference')
        saved = saved_models[0]
        observed = np.asarray(snapshot.session['_flow_cache'][1])
        # Preserve original float32 decode dtype before the archived NumPy speed conversion.
        decoded = np.asarray(saved['frames'][0]['raw_decoded'], dtype=np.float32)
        replay = first_window(trial, snapshot, topology, controls, decoded,
            executor_seed=920001, max_speed=config.simulation.vehicle_max_speed_mps)
        for bank in saved_models:
            assert_saved_reproduction(replay, bank)
        with torch.inference_mode():
            cpu_decoded = model.decode(model.encode(torch.as_tensor(observed, dtype=torch.float32).unsqueeze(0)))[0].numpy()
        codec_max = float(np.max(np.abs(cpu_decoded-decoded)))
        if not np.allclose(cpu_decoded, decoded, rtol=2e-5, atol=2e-5):
            raise ValueError('CPU codec differs materially from archived frozen GPU codec')
        reconstructed = first_window(trial, snapshot, topology, controls, cpu_decoded,
            executor_seed=920001, max_speed=config.simulation.vehicle_max_speed_mps)
        contrast = first_window(trial, snapshot, topology, controls, observed,
            executor_seed=920001, max_speed=config.simulation.vehicle_max_speed_mps)
        reals = [json.loads((directory/f'real/future_{seed}/continuation.json').read_text())
                 for seed in plan['future_seeds']]
        truth_costs = [value['targets']['1']['cost']['vehicle_seconds'] for value in reals]
        denominator = replay['cohort_size']*60.
        record = dict(condition=condition, seed=row['seed'], time=row['time'],
            cohort_size=replay['cohort_size'], saved_33_trajectories_exactly_reproduced=True,
            codec_cpu_vs_saved_max_abs=codec_max,
            cpu_vs_saved_event_contrast=compare_events(replay['own_action_events'], reconstructed['own_action_events']),
            saved_decoded_exposure=replay['vehicle_seconds'], cpu_codec_exposure=reconstructed['vehicle_seconds'],
            observed_flow_exposure=contrast['vehicle_seconds'], real_future_exposures=truth_costs,
            observed_minus_saved_vehicle_seconds=contrast['vehicle_seconds']-replay['vehicle_seconds'],
            saved_cost_mae_minutes=float(np.mean(np.abs(np.asarray(truth_costs)-replay['vehicle_seconds']))/denominator),
            observed_cost_mae_minutes=float(np.mean(np.abs(np.asarray(truth_costs)-contrast['vehicle_seconds']))/denominator),
            decoded_vs_observed_events=compare_events(replay['own_action_events'], contrast['own_action_events']),
            changed_vehicle_exposures=sum(replay['per_vehicle_exposure_seconds'][name] != value
                for name, value in contrast['per_vehicle_exposure_seconds'].items()))
        trial.atomic_json(args.output/f'{condition}.json', dict(record=record,
            saved_decoded=replay, cpu_codec=reconstructed, observed_flow=contrast))
        for path in [snapshot_path, directory/'protocol.json', *sorted((directory/'model').glob('*.json')),
                     *[directory/f'real/future_{seed}/continuation.json' for seed in plan['future_seeds']]]:
            artifacts[str(path.relative_to(ROOT))] = sha(path)
        records.append(record)
        print(json.dumps(record), flush=True)
    report = dict(checkpoint_sha256=EXPECTED_CHECKPOINT, source_manifest_sha256=EXPECTED_SOURCE,
        analysis_source_sha256=sha(__file__), device='cpu', cpu_threads=2, dynamics_transitions=0,
        sumo_started=False, gpu_used=False, training=False, records=records,
        balanced_saved_mae_minutes=float(np.mean([row['saved_cost_mae_minutes'] for row in records])),
        balanced_observed_mae_minutes=float(np.mean([row['observed_cost_mae_minutes'] for row in records])),
        limitation='Total initial-flow intervention also changes controller routing and chronological queue events. '
            'Observed flow is frozen for 30s; this is not true future speed, exact SUMO physics, or an isolated codec causal effect. '
            'Two initial seeds/four states are descriptive; no population superiority claim.', input_sha256=artifacts)
    trial.atomic_json(args.output/'report.json', report)
    lines = ['# Initial 30-second executor attribution', '',
        'Completed pretrained Prior+FM; frozen conditional-futures runtime; CPU two threads; no dynamics transition, SUMO, GPU, or training.', '',
        'All 33 archived model trajectories per state reproduce exactly in first-window cost, causal action events, and endpoint explicit cohort counts using their saved float32 initial reconstruction. Fresh CPU codec reconstructions are separately verified against those saved outputs.', '',
        '| State | C0 | Real cost range (vehicle-s) | Saved codec cost | CPU codec cost | Observed-flow cost | Saved MAE (min) | Observed MAE (min) | Next-hop changes / matched events |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        events = row['decoded_vs_observed_events']
        lines.append(f"| {row['condition']} | {row['cohort_size']} | {min(row['real_future_exposures']):.0f}–{max(row['real_future_exposures']):.0f} | {row['saved_decoded_exposure']:.0f} | {row['cpu_codec_exposure']:.0f} | {row['observed_flow_exposure']:.0f} | {row['saved_cost_mae_minutes']:.8f} | {row['observed_cost_mae_minutes']:.8f} | {events['next_hop_changes']} / {events['matched_occurrences']} |")
    lines += ['', f"Equal-state mean absolute cost errors: saved codec {report['balanced_saved_mae_minutes']:.8f} min; observed-flow executor {report['balanced_observed_mae_minutes']:.8f} min.", '',
        'Source noise enters the transition only after the first physical window, so the first-window cost is identically deterministic across all 32 sampled draws and the zero-noise draw. First-window errors cannot be attributed to sampled FM transition noise.', '', report['limitation'], '',
        'Detailed event alignments, per-vehicle exposure, controls, effective speeds, and original input SHA256 hashes are retained in the JSON artifacts.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
