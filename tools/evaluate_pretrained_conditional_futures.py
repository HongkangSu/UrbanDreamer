"""Feasibility of genuine Poisson futures after an identical past-only SUMO state.

No loadState, fabricated macrostate, or production-session changes are used.
Each replica reruns the identical past-only prefix, proves full boundary identity,
then registers its private future Poisson realization through the production
registration algorithm. The initial cohort alone defines count/cost targets.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import gzip
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools import evaluate_pretrained_decision_fidelity as fidelity
from tools.component_experiment_data import atomic_json, load_frozen, sha256
from urban_dreamer.learning.action_events import ActionEvent, pack_action_events
from urban_dreamer.traffic.demand import PoissonDemand, demand_identity
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.topology import VehicleDemand

HORIZONS = (1, 2, 4, 8)
VEHICLE_FIELDS = frozenset(('vehicle_id', 'current_edge', 'destination_edge', 'departure_time',
    'actual_departure_time', 'observed_departure_time', 'departed', 'arrived', 'remaining_distance',
    'committed_next_edge', 'pending_decision', 'in_junction', 'junction_remaining_distance',
    'junction_speed', 'junction_connection', 'arrival_time', 'visited_edges', 'junction_lane', 'junction_position'))


class GuardedController(fidelity.CausalShortestController):
    """Only production pending rows, current observed flow, and physical time."""
    def __init__(self, *args):
        super().__init__(*args)
        self.input_audit = dict(calls=0, vehicles=0, forbidden_input_fields=[], schedule_or_rate_access=False)

    def __call__(self, pending, flow, now):
        extra = set().union(*(set(row)-VEHICLE_FIELDS for row in pending)) if pending else set()
        if extra:
            raise ValueError(f'controller received non-observation fields: {sorted(extra)}')
        if np.asarray(flow).shape[0] != self.network.edge_count or not math.isfinite(now):
            raise ValueError('controller requires current road flow and finite physical time')
        self.input_audit['calls'] += 1
        self.input_audit['vehicles'] += len(pending)
        return super().__call__(pending, flow, now)


def past_only_demand(law, *, seed, boundary):
    """Use the actual source sampler, discarding all post-boundary registration."""
    if not 0 < boundary < law.duration_seconds:
        raise ValueError('boundary must lie inside the source demand interval')
    sampled = law.sample(seed=seed, episode_index=0)
    return tuple(vehicle for vehicle in sampled if vehicle.departure_second < boundary)


def sample_future(law, *, boundary, seed):
    """Homogeneous independent increments via the same PoissonDemand sampler."""
    remaining = float(law.duration_seconds-boundary)
    if remaining <= 0:
        raise ValueError('future interval must have positive source-derived duration')
    conditional = PoissonDemand(law.od_rates, remaining)
    before = fidelity.fingerprint(fidelity.external_rng_state())
    sampled = conditional.sample(seed=seed, episode_index=0)
    # Audit the source sampler's exact private generator before/after states.
    # The authoritative vehicle realization still comes from .sample above.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0, 137071]))
    initial_rng = copy.deepcopy(rng.bit_generator.state)
    arrivals, od_counts = [], []
    for origin, destination, rate in conditional.od_rates:
        count = int(rng.poisson(rate*remaining))
        times = rng.uniform(0., remaining, size=count) if count else ()
        arrivals.extend((float(t), origin, destination) for t in times)
        od_counts.append(dict(origin_index=origin, destination_index=destination, rate=rate, count=count))
    expected = [(time, origin, destination) for time, origin, destination in sorted(arrivals)]
    actual = [(row.departure_second, row.origin_index, row.destination_index) for row in sampled]
    if expected != actual:
        raise ValueError('private RNG audit no longer matches the source Poisson sampler')
    future = tuple(replace(row, vehicle_id=f'future_v{index}', departure_second=boundary+row.departure_second)
                   for index, row in enumerate(sampled))
    if any(not boundary < row.departure_second < law.duration_seconds for row in future):
        raise ValueError('sample must be strictly after capture and before source demand end')
    if fidelity.fingerprint(fidelity.external_rng_state()) != before:
        raise RuntimeError('future sampler consumed an external/global RNG stream')
    return future, dict(seed=seed, episode_index=0, boundary=boundary, source_demand_end=law.duration_seconds,
        remaining_duration=remaining, od_rates=[list(row) for row in law.od_rates], od_counts=od_counts,
        realized_count=len(future), realization_sha256=demand_identity(future),
        expected_count=conditional.expected_vehicle_count, generator=type(rng.bit_generator).__name__,
        initial_private_rng=initial_rng, final_private_rng=copy.deepcopy(rng.bit_generator.state),
        actual_source_sampler_used=True, global_rng_unchanged=True,
        naming='future_v{arrival ordinal}; names do not expose private seed or intensity',
        vehicles=[asdict(row) for row in future])


def register_future(session, vehicles, *, route_start=None):
    """Registration-only counterpart of SumoEventSession.reset's inline loop.

Route construction, vehicle.add arguments, and wrapper fields match production;
parity is tested against reset. Type creation/startup are intentionally omitted.
Subscriptions and departure accounting remain owned by production _observe.
"""
    if session._connection is None or session._closed or session._invalid:
        raise ValueError('future injection requires a valid open session')
    now = session._time_ms/1000.
    ordered = sorted(vehicles, key=lambda row:(row.departure_second,row.vehicle_id))
    names = [row.vehicle_id for row in ordered]
    if len(names) != len(set(names)) or set(names)&set(session._vehicles):
        raise ValueError('future vehicle names collide with existing or future registrations')
    if any(not math.isfinite(row.departure_second) or row.departure_second <= now for row in ordered):
        raise ValueError('future registration cannot inject at/before the capture time')
    first = len(session._vehicles) if route_start is None else route_start
    existing_routes = set(session._connection.route.getIDList())
    plan = []
    for index, demand in enumerate(ordered, start=first):
        origin = session.network.edge_to_index[session.topology.origins[demand.origin_index]]
        destination = session.network.edge_to_index[session.topology.destinations[demand.destination_index]]
        choices = session._legal_candidates(origin, destination)
        if choices:
            costs = session.candidates.costs(destination)
            initial = min(choices, key=lambda edge:(session.network.free_flow_travel_seconds[edge]+costs[edge],edge))
            route = session.candidates.full_route(origin, initial, destination)
        else:
            route = (origin,)
        route_id = f'urban_initial_{index}'
        if route_id in existing_routes:
            raise ValueError('future route ID collides with an existing route')
        plan.append((demand, origin, destination, route_id, route))
    records = []
    for demand, origin, destination, route_id, route in plan:
        edges = [session.network.edge_ids[edge] for edge in route]
        session._connection.route.add(route_id, edges)
        session._connection.vehicle.add(demand.vehicle_id, route_id, typeID='urban_cav',
            depart=str(demand.departure_second), departPos='0', departSpeed='0')
        session._vehicles[demand.vehicle_id] = dict(vehicle_id=demand.vehicle_id, current_edge=origin,
            destination_edge=destination, departure_time=float(demand.departure_second), actual_departure_time=None,
            observed_departure_time=None, departed=False, arrived=False,
            remaining_distance=float(session.network.lengths_m[origin]), committed_next_edge=None,
            pending_decision=False, in_junction=False, junction_remaining_distance=0., junction_speed=0.,
            junction_connection=None)
        if session.forbid_revisit:
            session._vehicles[demand.vehicle_id]['visited_edges'] = ()
            session._commanded_routes[demand.vehicle_id] = tuple(route)
            session._route_cursors[demand.vehicle_id] = -1
        records.append(dict(vehicle_id=demand.vehicle_id, route_id=route_id, edges=edges,
            depart=demand.departure_second, typeID='urban_cav', departPos='0', departSpeed='0'))
    # reset fills this cache only after registration. Refresh it here as well;
    # strictly future registration must leave current physical flow unchanged.
    session._flow_cache = None
    return records


def rng_xml_digest(path):
    with gzip.open(path, 'rb') as stream:
        root = ET.fromstring(stream.read())
    rng = [ET.tostring(row).decode() for row in root if 'rng' in row.tag.lower()]
    if not rng:
        raise ValueError('SUMO state must contain its RNG nodes')
    return fidelity.fingerprint(rng)


def current_physical(observation):
    return {key:observation[key] for key in ('time','vehicles','tls','edge_counts','edge_speeds','teleports')}


def cohort_frame(session, cohort, start):
    state = session._state()
    if session._connection.simulation.getTime() != state['time']:
        raise ValueError('SUMO and wrapper clocks disagree; no synthetic clock repair is permitted')
    counts = np.bincount([int(row['current_edge']) for row in state['vehicles'] if row['vehicle_id'] in cohort
        and row['departed'] and not row['arrived'] and not row.get('in_junction',False)], minlength=session.network.edge_count)
    cost = fidelity.fixed_cohort_cost(state['vehicles'], cohort, start, state['time']) if state['time']>start else dict(
        vehicle_seconds=0.,mean_minutes=0.,cohort_size=len(cohort),arrivals=0,censored_remaining_time=True,timeout_penalty=0.)
    return dict(time=state['time'], relative_time=state['time']-start, cohort_counts=counts.tolist(),
        cohort_active=sum(not session._vehicles[name]['arrived'] for name in cohort),
        full_observed_flow=np.asarray(state['flow']).tolist(), cost=cost,
        future_departed=sum(row['departed'] for name,row in session._vehicles.items() if name.startswith('future_v')),
        vehicles=copy.deepcopy(state['vehicles']))


def future_continuation(session, controller, cohort, boundary):
    frames = [cohort_frame(session,cohort,boundary)]
    windows = []
    for _ in range(9):
        controller.prepare(session._flow(),session._time_ms/1000.)
        result = session.advance_window(controller)
        frames.append(cohort_frame(session,cohort,boundary))
        windows.append(dict(start=result.start_time,end=result.end_time,decisions=[asdict(row) for row in result.decisions]))
    speed_limits = np.asarray(session.network.speed_limits_mps)
    targets = {}
    for h in HORIZONS:
        current, following = np.asarray(frames[h]['full_observed_flow']),np.asarray(frames[h+1]['full_observed_flow'])
        targets[str(h)] = dict(horizon_windows=h,time=frames[h]['time'],
            count_target_cohort=frames[h]['cohort_counts'], explicit_cohort_active=frames[h]['cohort_active'],
            speed_target_mps=(.5*(current[:,1]+following[:,1])*speed_limits).tolist(),
            speed_label_times=[frames[h]['time'],frames[h+1]['time']], cost=frames[h]['cost'])
    return dict(frames=frames,windows=windows,targets=targets,controller_input_audit=controller.input_audit,
        label_contract='same-frame C0 count; speed is mean of actual H/H+1 boundary speeds, including background effects')


def normalization_probe(campaign, snapshot, topology, config):
    """Small CPU check on the complete pretrained event-protocol model only."""
    _,_,_,model,_ = load_frozen(campaign,device='cpu')
    model.eval().requires_grad_(False)
    if not model.dynamics.event_protocol:
        raise ValueError('normalization invariance requires the actual event-protocol checkpoint')
    pending = [snapshot.session['_vehicles'][name] for name in sorted(snapshot.session['_pending'])]
    if not pending:
        raise ValueError('nonempty legal event fixture required for normalization probe')
    candidates = fidelity.NextHopCandidates(topology.network,traffic_model='meso')
    controller = fidelity.CausalShortestController(topology.network,candidates,config.simulation.vehicle_max_speed_mps)
    flow = snapshot.session['_flow_cache'][1]
    start = snapshot.session['_time_ms']/1000.
    controller.prepare(flow,start)
    decision = controller(pending[:1],flow,start)[0]
    event = ActionEvent(decision.vehicle_id,0,start,decision.current_edge,decision.next_edge,pending[0]['destination_edge'])
    events = pack_action_events([[(event,)]],window_starts=[[start]])
    population = len(snapshot.session['_vehicles'])
    action = torch.zeros((1,topology.network.edge_count));action[0,decision.next_edge]=1.
    first,second = action/population,action/(population*3)
    with torch.inference_mode():
        latent = model.encode(torch.as_tensor(flow).unsqueeze(0))
        noise = torch.zeros((1,model.dynamics.graph_prior.rank,64))
        a = model.dynamics.encode_condition(first.unsqueeze(1),action_events=events)
        b = model.dynamics.encode_condition(second.unsqueeze(1),action_events=events)
        next_a = model.transition(latent,first,action_events=events,noise=noise,steps=2,solver='euler')
        next_b = model.transition(latent,second,action_events=events,noise=noise,steps=2,solver='euler')
    if not torch.equal(a.edge_tokens,b.edge_tokens) or not torch.equal(next_a,next_b):
        raise ValueError('numeric action normalization changed event-protocol prediction')
    return dict(event_protocol=True,complete_pretrained_checkpoint=fidelity.PRETRAINED_SHA256,
        device='cpu',solver_steps=2,purpose='invariance check only, not a performance comparison',
        denominators=[population,population*3],nonempty_events=len(events.vehicle_ids),
        edge_condition_exactly_equal=True,next_latent_exactly_equal=True,max_abs_difference=float((next_a-next_b).abs().max()),
        interpretation='with fixed sparse events, numeric action values are shape-checked but unused by event conditioning; fixed known past-population normalization is causal')


def model_rollout(model, snapshot, topology, controls, *, draw_seed, steps=100,
                  executor_seed=0, mode='sampled_prior_fm', max_speed=40.):
    """Own causal C0 rollout; intentionally accepts no future-demand realization.

    Reuses the decision-fidelity executor and controller, with no intervention.
    Raw decoder predictions remain separate from the executor's explicit counts.
    """
    if mode not in ('sampled_prior_fm', 'zero_noise_prior_fm') or steps < 1:
        raise ValueError('invalid conditional model interpretation or solver budget')
    if not model.dynamics.event_protocol or model.training:
        raise ValueError('conditional model rollout requires evaluated sparse-event dynamics')
    before_rng = fidelity.fingerprint(fidelity.external_rng_state())
    network = topology.network
    device = next(model.parameters()).device
    candidates = fidelity.NextHopCandidates(network, traffic_model='meso')
    controller = GuardedController(network, candidates, max_speed)
    states = [copy.deepcopy(row) for row in snapshot.session['_vehicles'].values()
              if row['departed'] and not row['arrived']]
    if not states:
        raise ValueError('conditional model rollout requires a nonempty initial cohort')
    population = len(snapshot.session['_vehicles'])  # Known past only, before injection.
    start = snapshot.session['_time_ms']/1000.
    flow = snapshot.session['_flow_cache'][1]
    generator = torch.Generator(device='cpu').manual_seed(draw_seed)
    arbiter = fidelity.ChronologicalJunctionArbiter(controls, seed=executor_seed, uncertain_order=True)

    def decide(pending, current, now):
        # Private executor queue bookkeeping is not a controller observation.
        rows = [{key:value for key,value in row.items() if not key.startswith('_city_')} for row in pending]
        return controller(rows, current, now)

    capacity = np.maximum(np.asarray(network.capacities), 1.)
    limits = np.maximum(np.asarray(network.speed_limits_mps), .1)
    frames, windows, seconds = [], [], 0.
    with torch.inference_mode():
        latent = model.encode(torch.as_tensor(flow, dtype=torch.float32, device=device).unsqueeze(0))
        for index in range(max(HORIZONS)+1):
            now = start+30*index
            decoded = model.decode(latent)[0].cpu().numpy()
            explicit = np.bincount([int(row['current_edge']) for row in states
                if not row['arrived'] and not row.get('in_junction',False)], minlength=network.edge_count)
            raw_count = decoded[:,2]*capacity
            frames.append(dict(horizon_windows=index,time=now,raw_decoded=decoded.tolist(),
                raw_decoded_count=raw_count.tolist(),raw_decoded_speed_mps=(decoded[:,1]*limits).tolist(),
                explicit_cohort_count=explicit.tolist(),explicit_cohort_active=sum(not row['arrived'] for row in states),
                raw_count_total=float(raw_count.sum()),explicit_count_total=int(explicit.sum()),
                decoded_minus_explicit_count_total=float(raw_count.sum()-explicit.sum()),
                vehicle_seconds=seconds,mean_minutes=seconds/len(states)/60.))
            if index == max(HORIZONS):
                break
            controller.prepare(decoded,now)
            speed = np.minimum(fidelity.clip_road_speed_fractions(decoded[:,1],'meso')*limits,max_speed)
            physical = fidelity.advance_city_imagined_window(states,candidates=candidates,edge_speeds_mps=speed,
                frozen_flow=decoded,start_time=now,total_vehicle_count=population,decide=decide,
                junction_arbiter=arbiter,interval_seconds=30.,observation_step_seconds=1.,
                session_protocol='event',traffic_model='meso',vehicle_max_speed_mps=max_speed,stop_when_empty=False)
            events = pack_action_events([[physical.action_events]],window_starts=[[now]])
            noise = torch.zeros((1,model.dynamics.graph_prior.rank,64))
            if mode == 'sampled_prior_fm':
                noise = torch.randn(noise.shape,generator=generator)
            latent = model.transition(latent,torch.as_tensor(physical.action,device=device).unsqueeze(0),
                action_events=events.to(device),noise=noise.to(device),steps=steps,solver='euler')
            seconds += physical.active_vehicle_seconds
            windows.append(dict(start=now,end=physical.end_time,vehicle_seconds=physical.active_vehicle_seconds,
                own_action_events=[asdict(event) for event in physical.action_events]))
    if fidelity.fingerprint(fidelity.external_rng_state()) != before_rng:
        raise RuntimeError('model rollout consumed the shared external RNG stream')
    return dict(mode=mode,draw_seed=draw_seed,executor_seed=executor_seed,solver_steps=steps,
        cohort_size=len(states),known_past_normalization=population,frames=frames,windows=windows,
        targets={str(h):frames[h] for h in HORIZONS},controller_input_audit=controller.input_audit,
        executor_audit=arbiter.audit,external_rng_unchanged=True,
        queue_order='spatial-order approximation; tied positions use declared private seeded uncertain order',
        limitation='C0-only operational executor; no explicit future births, despite background-sensitive decoder speed targets',
        conditioning='real replicas fix full microstate; FM conditions on aggregate observations and own causal events')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--condition-seed',type=int,default=620001)
    parser.add_argument('--boundary',type=int,default=300)
    parser.add_argument('--future-seeds',type=int,nargs='+',default=[810001,810002])
    parser.add_argument('--normalization-probe',action='store_true')
    args=parser.parse_args(argv)
    if args.boundary not in (300,900) or args.condition_seed not in (620001,620002):
        parser.error('use the four predeclared initial conditions only')
    if len(args.future_seeds)<2 or len(set(args.future_seeds))!=len(args.future_seeds):
        parser.error('at least two distinct private future seeds are required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    source=fidelity.validate_pretrained(args.campaign)
    config=fidelity.load_config(args.campaign/'snapshot/run_config.toml')
    prepared=fidelity.validate_prepared_inputs(config)
    topology=fidelity.load_traffic_topology(prepared)
    law=PoissonDemand.from_vehicles(topology.vehicles,duration_seconds=config.simulation.demand_duration_seconds)
    past=past_only_demand(law,seed=args.condition_seed,boundary=args.boundary)
    if not past:
        raise ValueError('prescribed prefix is empty; no replacement condition allowed')
    protocol=dict(schema_version=1,source=source,input_hashes=prepared.input_hashes,
        condition_seed=args.condition_seed,boundary=args.boundary,
        future_seeds=args.future_seeds,source_demand_duration=law.duration_seconds,od_rates=[list(row) for row in law.od_rates],
        past_count=len(past),past_realization_sha256=demand_identity(past),horizons=list(HORIZONS),following_capture_seconds=270,
        recovery='identical past-only prefix replay; genuine independent future-demand inputs registered after the full-state gate',
        controller_inputs='pending production vehicle rows/current observed flow/physical time only; no seed/schedule/intensity/population fields',
        count_cost_scope='fixed initial active cohort C0; background births excluded from counts/cost but retained in traffic dynamics and speed labels',
        speed_label='.5*(observed normalized speed at H + actual following boundary H+1) times road speed limit',
        model_comparison_launched=False,model_limitation='operational model currently advances C0 only; it does not instantiate sampled future background births',
        known_model_normalization='registered past count only; no realized future count is supplied to a model or controller',
        script_sha256=sha256(__file__),session_source_sha256=sha256(Path(inspect.getfile(SumoEventSession))),
        poisson_source_sha256=sha256(Path(inspect.getfile(PoissonDemand))),
        registration_semantics='parity-tested registration-only copy of reset loop; production routing/vehicle.add/_observe reused')
    bundle_path=ROOT/'manifest.json'
    if bundle_path.is_file():
        bundle=json.loads(bundle_path.read_text())
        if any(sha256(ROOT/name)!=value for name,value in bundle['source_sha256'].items()):
            raise ValueError('frozen conditional-futures source bundle changed')
        protocol.update(source_bundle=str(ROOT),source_bundle_manifest_sha256=sha256(bundle_path))
    atomic_json(args.output/'protocol.json',protocol)
    prefix_rng=fidelity.external_rng_state()
    reference=None; records=[]
    branch_config=replace(config,run=replace(config.run,seed=args.condition_seed,device='cpu',output_dir=args.output),
        simulation=replace(config.simulation,sumo_extra_args=tuple(config.simulation.sumo_extra_args)+
            ('--save-state.rng','true','--save-state.precision','17')))
    for future_seed in args.future_seeds:
        output=args.output/f'future_{future_seed}';output.mkdir()
        fidelity.restore_external_rng(prefix_rng)
        with SumoEventSession(branch_config,prepared,replace(topology,vehicles=past),backend='libsumo') as session:
            session.reset()
            controller=GuardedController(topology.network,session.candidates,config.simulation.vehicle_max_speed_mps)
            while session._time_ms<args.boundary*1000:
                controller.prepare(session._flow(),session._time_ms/1000.)
                session.advance_window(controller)
            if session._connection.simulation.getTime()!=args.boundary:
                raise ValueError('past-only prefix did not physically reach the declared boundary')
            controller.prepare(session._flow(),args.boundary)
            snapshot=fidelity.capture_state(session,controller,output/'before_registration.xml.gz',demand=dict(
                initial_seed=args.condition_seed,past_realization_sha256=demand_identity(past),future_inputs_registered=False))
            if any(row['departure_time']>=args.boundary and not row['departed'] for row in snapshot.session['_vehicles'].values()):
                raise ValueError('future vehicle was present before external-input registration')
            identity=dict(sumo_sha256=snapshot.sumo_sha256,python_controller_rng_sha256=snapshot.causal_sha256,
                observation_sha256=fidelity.fingerprint(snapshot.observation),controller_guard_sha256=fidelity.fingerprint(controller.input_audit))
            if reference is None:
                reference=snapshot;expected_identity=identity
                torch.save(snapshot.__dict__,args.output/'initial_python.pt')
            if identity!=expected_identity:
                atomic_json(output/'boundary_failure.json',dict(expected=expected_identity,actual=identity))
                raise ValueError('future replicas do not share the exact full initial state')
            atomic_json(output/'initial_identity.json',dict(identity,exact_match=True,before_future_registration=True))
            future,schedule=sample_future(law,boundary=args.boundary,seed=future_seed)
            atomic_json(output/'future_schedule.json',fidelity.plain(schedule))
            before_controller=fidelity.fingerprint(controller.state_dict())
            before_counters=fidelity.fingerprint(dict(metrics=session.metrics,reward=session.reward_accounting,
                active=session._active,pending=session._pending))
            registration=register_future(session,future)
            after=fidelity.simulator_observation(session)
            fidelity.save_sumo(session,output/'after_registration.xml.gz')
            injection=dict(registered=len(registration),records=registration,
                active_positions_routes_TLS_clock_counts_unchanged=fidelity.fingerprint(current_physical(after))==
                    fidelity.fingerprint(current_physical(snapshot.observation)),
                controller_unchanged=fidelity.fingerprint(controller.state_dict())==before_controller,
                counters_unchanged=fidelity.fingerprint(dict(metrics=session.metrics,reward=session.reward_accounting,
                    active=session._active,pending=session._pending))==before_counters,
                simulator_rng_unchanged=rng_xml_digest(output/'before_registration.xml.gz')==rng_xml_digest(output/'after_registration.xml.gz'),
                future_registered_but_not_started=sum(not session._vehicles[row.vehicle_id]['departed'] for row in future),
                full_state_after_injection_expected_to_differ=True)
            atomic_json(output/'registration_audit.json',injection)
            if not all(injection[key] for key in ('active_positions_routes_TLS_clock_counts_unchanged','controller_unchanged','counters_unchanged')):
                raise ValueError('future registration changed the pre-existing physical state or controller')
            cohort=set(snapshot.session['_active'])
            continuation=future_continuation(session,controller,cohort,args.boundary)
            atomic_json(output/'continuation.json',fidelity.plain(continuation))
            records.append(dict(future_seed=future_seed,future_count=len(future),future_sha256=schedule['realization_sha256'],
                initial_identity=identity,registration_simulator_rng_unchanged=injection['simulator_rng_unchanged'],
                horizon_costs={h:row['cost'] for h,row in continuation['targets'].items()},
                final_future_departed=continuation['frames'][-1]['future_departed']))
    if len({row['future_sha256'] for row in records})!=len(records):
        raise ValueError('private future seeds did not produce distinct schedules')
    if args.normalization_probe:
        atomic_json(args.output/'normalization_probe.json',normalization_probe(args.campaign,reference,topology,config))
    atomic_json(args.output/'feasibility.json',dict(feasible=True,initial_state_identical_before_future_input=True,
        replicas=records,controller_future_information_access=False,model_comparison_launched=False,
        note='new past-only-prefix conditional experiment; not a restoration of prior whole-future-registered campaign states'))
    print(json.dumps(dict(feasible=True,replicas=len(records),future_counts=[row['future_count'] for row in records])),flush=True)


if __name__=='__main__':
    main()
