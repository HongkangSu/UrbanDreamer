"""Paired real-SUMO evaluation of causal shortest-time and paper GA routing.

Policies see current measured edge travel times and the current vehicle's OD,
never the generated future demand table. Production runs/checkpoints are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city', choices=('ingolstadt_custom', 'coulommiers', 'montereau_fault_yonne'), default='ingolstadt_custom')
    parser.add_argument('--reference', type=Path, help='Optional paired evaluation JSON to verify input and demand hashes')
    parser.add_argument('--method', choices=('realtime_shortest', 'ga_routing', 'frozen_dta'), required=True)
    parser.add_argument('--demand-profile', choices=('original', 'peaks', 'od_stagger', 'burst'), default='original')
    parser.add_argument('--cost-observation-seconds', type=int, default=1,
                        help='Shortest-route road costs sampled on a global physical clock; decisions remain event-driven')
    parser.add_argument('--assignment', type=Path)
    parser.add_argument('--assignment-audit', type=Path, help='Input-hash and checkpoint audit for frozen DTA')
    parser.add_argument('--source', type=Path, default=REPO/'outputs/city_baselines_paper_reward_20261010/sources/ingolstadt_custom')
    parser.add_argument('--output', type=Path, default=REPO/'outputs/ingo_causal_routing_20261010')
    parser.add_argument('--ga-variant', choices=('author', 'feasible'), default='author')
    parser.add_argument('--ga-generations', type=int, default=50)
    parser.add_argument('--ga-source', type=Path, default=REPO/'outputs/ingo_causal_routing_20261010/references/author_repo/src/baselines/genetic_algorithm.py')
    parser.add_argument('--seeds', type=int, nargs='+', default=[10042, 10043, 10044])
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.ga_generations < 1:
        parser.error('--ga-generations must be positive')
    if args.cost_observation_seconds < 1:
        parser.error('--cost-observation-seconds must be positive')
    if args.cost_observation_seconds != 1 and args.method != 'realtime_shortest':
        parser.error('Restricted road-cost observation is supported only for realtime_shortest')
    if args.method == 'frozen_dta' and (args.assignment is None or args.assignment_audit is None):
        parser.error('frozen_dta requires --assignment and --assignment-audit')
    args.source = args.source.resolve(); args.output = args.output.resolve()
    sys.path.insert(0, str(args.source))
    import torch
    from urban_dreamer.learning.baselines.common import EpisodeRunner
    from urban_dreamer.traffic.event_session import SumoEventSession
    import libsumo
    module_path = REPO/'tools/causal_routing_policies.py'
    spec = importlib.util.spec_from_file_location('causal_eval_policies', module_path)
    policies = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = policies
    spec.loader.exec_module(policies)
    torch.set_num_threads(1)
    output = args.output/args.method
    output.mkdir(parents=True, exist_ok=False)
    runargs = argparse.Namespace(config=args.source/'configs'/f'{args.city}.toml',
        device='cpu', threads=1, seed=42, vehicle_limit=0, session='event',
        backend='libsumo', step_seconds=1., forbid_revisit=False, sp_prior_weight=0.,
        global_reward_only=True, reward_objective='observed_mean')
    handle = {}
    session_type = SumoEventSession
    observer_path = REPO/'tools/routing_cost_observer.py'
    if args.cost_observation_seconds > 1:
        observer_spec = importlib.util.spec_from_file_location('causal_cost_observer', observer_path)
        observer_module = importlib.util.module_from_spec(observer_spec)
        observer_spec.loader.exec_module(observer_module)

        class ObservationLimitedSession(SumoEventSession):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.routing_cost_observer = observer_module.PeriodicCostObserver(
                    args.cost_observation_seconds, self.network.free_flow_travel_seconds)

            def _read_routing_costs(self):
                return [self._connection.edge.getTraveltime(edge) for edge in self.network.edge_ids]

            def reset(self):
                state = super().reset()
                self.routing_cost_observer.observe(0, self._read_routing_costs)
                return state

            def _apply_decisions(self, callback, frozen_flow):
                # Observe even if there are no pending decisions at this tick.
                self.routing_cost_observer.observe(self._time_ms / 1000., self._read_routing_costs)
                return super()._apply_decisions(callback, frozen_flow)

        session_type = ObservationLimitedSession
    def factory(*a, **kw):
        handle['session'] = session_type(*a, **kw, step_seconds=1., backend='libsumo')
        return handle['session']
    runner = EpisodeRunner(runargs, session_factory=factory)
    profile_path = REPO/'tools/routing_demand_profiles.py'
    profile_spec = importlib.util.spec_from_file_location('causal_eval_demand_profiles', profile_path)
    profiles = importlib.util.module_from_spec(profile_spec)
    sys.modules[profile_spec.name] = profiles
    profile_spec.loader.exec_module(profiles)
    demand_contract = profiles.profile_metadata(args.demand_profile, runner.demand.duration_seconds)
    if args.demand_profile != 'original':
        runner.demand = profiles.ReTimedDemand(runner.demand, args.demand_profile)
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference is not None:
        assert reference['input_hashes'] == runner.prepared.input_hashes
    assert runner.deadline == 3810
    author_path = args.ga_source.resolve()
    improved_path = REPO/'tools/ga_feasible_routing.py'
    assignment_contract = None
    if args.method == 'frozen_dta':
        from urban_dreamer.learning.baselines.dta import DtaMethod
        assignment_hash = hashlib.sha256(args.assignment.read_bytes()).hexdigest()
        audit = json.loads(args.assignment_audit.read_text())
        assignment_contract = next(entry for entry in audit['references'] if entry['city'] == args.city)
        assert assignment_contract['checkpoint_sha256'] == assignment_hash
        assert assignment_contract['input_hashes'] == runner.prepared.input_hashes
        policy = DtaMethod(runargs, runner)
        policy.load_state_dict(torch.load(args.assignment, map_location='cpu', weights_only=False))
        assert policy.iteration == assignment_contract['iteration']
    elif args.method == 'realtime_shortest':
        policy = policies.RealTimeShortestRouter(runner.network)
    elif args.ga_variant == 'author':
        policy = policies.AuthorGeneticRouter(runner.network, author_path, generations=args.ga_generations)
    else:
        sys.path.insert(0, str(REPO))
        from tools.ga_feasible_routing import FeasibleGeneticRouter
        policy = FeasibleGeneticRouter(runner.network, author_path, generations=args.ga_generations)
    identity = dict(city=args.city, expected_vehicle_count=runner.demand.expected_vehicle_count, network_edges=runner.network.edge_count, method=args.method, source=str(args.source), input_hashes=runner.prepared.input_hashes,
        simulation=runner.simulation_settings, sumo_version=libsumo.getVersion(),
        seeds=args.seeds, demand_seed=42, deadline_seconds=runner.deadline,
        observation_seconds=1., decision_semantics='same branch-only event session as mainline',
        cost_observation_seconds=args.cost_observation_seconds,
        cost_observation_clock=('global physical times 0, period, 2*period; frozen between ticks'
            if args.cost_observation_seconds > 1 else 'current data at each observed decision timestamp'),
        cost_observer_source_sha256=(hashlib.sha256(observer_path.read_bytes()).hexdigest()
            if args.cost_observation_seconds > 1 else None),
        cost=('frozen historically learned marginal costs and OD path fractions' if args.method == 'frozen_dta'
              else 'current libsumo.edge.getTraveltime; invalid/nonpositive values use static free-flow'),
        observation_access=('current vehicle position/destination and fixed historical assignment; no test-demand optimization'
            if args.method == 'frozen_dta' else 'last periodic full-network cost snapshot; live vehicle position/destination and legal candidates'
            if args.cost_observation_seconds > 1 else 'all current edge travel times; only current vehicle position/destination and legal candidates'),
        future_demand_access=False, final_population_access=False, optimizer_updates=0,
        demand_profile=demand_contract, assignment=assignment_contract,
        demand_profile_source_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        policy_source_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
        actor_baseline_difference=('Historical assignment does not refresh learned costs within evaluation.' if args.method == 'frozen_dta'
            else 'Road-cost refresh period is explicit; full-network shortest-path planning remains different from actor features.' if args.cost_observation_seconds > 1
            else 'Road costs refreshed at each observed decision timestamp; earlier RL used window-frozen road features.'),
        ga=(dict(variant=args.ga_variant, variant_source_sha256=(hashlib.sha256(improved_path.read_bytes()).hexdigest() if args.ga_variant=='feasible' else None), population=100, generations=args.ga_generations, crossover=.8, mutation=.1, tournament=5,
            cache_ttl_seconds=300, cache_rule='per current-edge/destination entry age, not global synchronized updates',
            author_source=str(author_path),author_sha256=hashlib.sha256(author_path.read_bytes()).hexdigest(),
            adaptations=['SUMO legal-turn edge graph','actual simulation seconds advance cache clock','reset cache and random seed per episode'],
            preserved=('original GA operators, random-walk initialization, convergence rule and first-legal fallback' if args.ga_variant=='author' else 'original fitness/evolution/cache; reachable static-potential route construction replaces random walks'))
            if args.method=='ga_routing' else None))
    write_json(output/'protocol.json', identity)
    records=[]; run_start=time.monotonic()
    def archive_metadata(metadata):
        metadata['routing_cost_observation_seconds'] = args.cost_observation_seconds
        if args.demand_profile != 'original':
            metadata['demand']['law'] = demand_contract['demand_law']
            metadata['demand']['od_rates_interpretation'] = 'period-average; temporal multipliers in timing_profile'
            metadata['demand']['timing_profile'] = demand_contract
            metadata['experiment_only'] = True
        return metadata

    for seed in args.seeds:
        if args.method == 'frozen_dta':
            policy.stats = dict(decision_count=0, assignment_iteration=policy.iteration)
        else:
            policy.reset()
        random.seed(seed); np.random.seed(seed)
        last_timestamp=None; query_seconds=selection_seconds=0.; query_count=invalid_costs=0
        progress_bucket=-1
        def select(vehicles, actor_obs, frozen_flow, now):
            nonlocal last_timestamp,query_seconds,selection_seconds,query_count,invalid_costs,progress_bucket
            if args.method == 'frozen_dta':
                tick = time.monotonic()
                choices = policy.select(vehicles, actor_obs, frozen_flow, now, explore=False)
                selection_seconds += time.monotonic() - tick
                policy.stats['decision_count'] += len(vehicles)
                return choices
            if args.cost_observation_seconds > 1:
                observer = handle['session'].routing_cost_observer
                if last_timestamp != observer.timestamp:
                    policy.update_costs(observer.values, observer.timestamp)
                    last_timestamp = observer.timestamp
                query_seconds = observer.query_seconds
                query_count = observer.query_count
                invalid_costs = observer.invalid_cost_count
            elif last_timestamp != now:
                tick=time.monotonic();api=handle['session']._connection.edge
                costs=np.asarray([api.getTraveltime(e) for e in runner.network.edge_ids],dtype=np.float64)
                invalid=~np.isfinite(costs)|(costs<=0)
                invalid_costs+=int(invalid.sum())
                costs[invalid]=runner.network.free_flow_travel_seconds[invalid]
                policy.update_costs(costs,now)
                query_seconds+=time.monotonic()-tick; query_count+=1; last_timestamp=now
            tick=time.monotonic();choices=[]
            for row,vehicle in enumerate(vehicles):
                cols=np.flatnonzero(actor_obs.mask[row]);legal=actor_obs.edge_indices[row,cols].tolist()
                nxt=policy.next_edge(int(vehicle['current_edge']),int(vehicle['destination_edge']),legal)
                choices.append(int(cols[legal.index(nxt)]))
            selection_seconds+=time.monotonic()-tick
            bucket=int(now//300)
            if bucket!=progress_bucket:
                progress_bucket=bucket
                print(json.dumps(dict(method=args.method,seed=seed,time=now,policy_stats=policy.stats,
                    cost_query_seconds=query_seconds,selection_seconds=selection_seconds)),flush=True)
            return choices,[dict(entropy=0.) for _ in choices]
        _,_,metrics=runner.run_episode(select,seed=seed,label=args.method,
            archive_path=output/f'evaluation_{seed}.npz',archive_metadata=dict(phase='evaluation',episode=seed),
            metadata_transform=archive_metadata,
            on_demand=(policy.begin_episode if args.method == 'frozen_dta' else None))
        if args.method == 'frozen_dta':
            assert policy.iteration == assignment_contract['iteration']
        sample_times = None
        if args.cost_observation_seconds > 1:
            observer = handle['session'].routing_cost_observer
            query_seconds, query_count = observer.query_seconds, observer.query_count
            invalid_costs = observer.invalid_cost_count
            sample_times = observer.sample_times
        if reference is not None:
            ref=next(r for r in reference['evaluations'] if r['seed']==seed)
            assert metrics['demand_sha256']==ref['demand_sha256'] and metrics['vehicle_count']==ref['vehicle_count']
        row=dict(seed=seed,metrics=metrics,policy_stats=dict(policy.stats),cost_query_seconds=query_seconds,
            selection_seconds=selection_seconds,cost_query_count=query_count,invalid_cost_fallback_count=invalid_costs,
            cost_observation_times=sample_times,
            archive=str(output/f'evaluation_{seed}.npz'))
        records.append(row)
        write_json(output/'result.json',dict(protocol=identity,rows=records,status='completed' if len(records)==len(args.seeds) else 'running',elapsed_seconds=time.monotonic()-run_start))
        print(json.dumps(dict(method=args.method,seed=seed,ATT_seconds=metrics['observed_mean_seconds'],
            completion=metrics['completion_rate'],collection_seconds=metrics['collection_seconds'],policy_stats=policy.stats)),flush=True)
    assert hashlib.sha256(module_path.read_bytes()).hexdigest()==identity['policy_source_sha256']
    assert hashlib.sha256(profile_path.read_bytes()).hexdigest()==identity['demand_profile_source_sha256']
    if args.cost_observation_seconds > 1:
        assert hashlib.sha256(observer_path.read_bytes()).hexdigest()==identity['cost_observer_source_sha256']


if __name__ == '__main__':
    main()
