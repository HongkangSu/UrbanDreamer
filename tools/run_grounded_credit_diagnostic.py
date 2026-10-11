"""Experimental real per-vehicle credit fit; never a strict production resume.

Labels use future realized outcomes; every critic feature is observed at the
decision. Unexecuted candidate values remain function-approximation extrapolation.
"""
from __future__ import annotations
import argparse, copy, json, math, random, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_online_pilot import atomic_json, sha256_file
from tools.run_real_feedback_ablation import aggregate
from tools.run_entropy_ablation import tree_hash
from urban_dreamer.models.policy import NextHopActor
from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.traffic.demand import PoissonDemand, demand_identity
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology, NextHopCandidates
from urban_dreamer.workflow.config import load_config


class RealCostCritic(nn.Module):
    """Shared factual-action regression and failure prediction, no edge identity."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(15, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 2))

    def forward(self, features):
        value = self.net(features)
        # Positive travel-time output; free-flow cost is an observed input only.
        return torch.nn.functional.softplus(value[..., 0]), value[..., 1]


def make_adapter(source, output):
    cfg = load_config(source / 'run_config.toml')
    prepared = validate_prepared_inputs(cfg)
    topology = load_traffic_topology(prepared)
    adapter = MainlineTrainer.__new__(MainlineTrainer)
    adapter.device = torch.device('cpu')
    adapter.config = replace(cfg, run=replace(cfg.run, output_dir=output, device='cpu', seed=314159))
    adapter.prepared_inputs, adapter.topology, adapter.network = prepared, topology, topology.network
    adapter.candidates = NextHopCandidates(topology.network)
    adapter.demand = PoissonDemand.from_vehicles(topology.vehicles,
        duration_seconds=cfg.simulation.demand_duration_seconds)
    adapter.actor = NextHopActor()
    adapter._session_factory = None
    return adapter


def collect(adapter, output, indices, capture=False):
    original = adapter._select
    rows = []
    for index in indices:
        features, masks, actions, times, vehicles, currents, destinations = [], [], [], [], [], [], []
        def hooked(states, flow, timestamp, *, context, mode, actor=None, observations=None):
            decisions = original(states, flow, timestamp, context=context, mode=mode,
                actor=actor, observations=observations)
            if capture:
                assert observations is not None
                current = np.asarray([s['current_edge'] for s in states], dtype=np.int64)
                local = np.asarray(flow[current, :4], dtype=np.float32)
                extra = np.concatenate((local, np.full((len(states), 1), timestamp / 3810, np.float32)), axis=1)
                features.append(np.concatenate((observations.features, np.broadcast_to(extra[:, None],
                    (*observations.mask.shape, 5))), axis=-1).copy())
                masks.append(observations.mask.copy())
                actions.extend(int(np.flatnonzero(observations.edge_indices[i] == decision.next_edge)[0])
                    for i, decision in enumerate(decisions))
                times.extend([timestamp] * len(states))
                vehicles.extend(str(s['vehicle_id']) for s in states)
                currents.extend(current.tolist())
                destinations.extend(int(s['destination_edge']) for s in states)
            return decisions
        adapter._select = hooked
        adapter.config = replace(adapter.config, run=replace(adapter.config.run, output_dir=output))
        adapter._next_environment_episode = index
        random.seed(271828 + index); np.random.seed(271828 + index); torch.manual_seed(271828 + index)
        started = time.monotonic()
        episode = adapter._run_episode(mode='online', episode_index=index)
        row = dict(index=index, demand_hash=demand_identity(adapter.demand.sample(seed=314159,
            episode_index=index)), population=episode.total_vehicle_count, metrics=episode.metrics,
            wall_seconds=time.monotonic() - started)
        rows.append(row); atomic_json(output / 'results.json', rows)
        if capture:
            vid = np.asarray(vehicles)
            timestamp = np.asarray(times, dtype=np.float32)
            failures = np.asarray([v not in episode.arrival_times for v in vehicles], np.float32)
            endpoint = np.asarray([episode.arrival_times.get(v, adapter._episode_end_time('online'))
                for v in vehicles], np.float32)
            label = np.maximum(endpoint - timestamp, 0) / 60
            directory = output / 'credit'; directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / f'episode_{index:03d}.npz', features=np.concatenate(features),
                mask=np.concatenate(masks), action=np.asarray(actions, np.int64), time=timestamp,
                vehicle=vid, current=np.asarray(currents, np.int64), destination=np.asarray(destinations, np.int64),
                remaining_minutes=label, failure=failures, demand_index=np.full(len(label), index, np.int64))
        adapter._select = original
        print(json.dumps(dict(stage='collection', directory=str(output), **row)), flush=True)
    return rows


def critic_metrics(critic, x, y, f, device):
    with torch.no_grad():
        predictions = [critic(x[i:i + 4096].to(device)) for i in range(0, len(x), 4096)]
        t = torch.cat([r[0].cpu() for r in predictions]) * 60
        p = torch.cat([r[1].sigmoid().cpu() for r in predictions])
    residual = y - t
    bins = []
    for lower in (0., .2, .4, .6, .8):
        selected = (p >= lower) & (p < lower + .2 + 1e-7)
        if selected.any():
            bins.append(dict(lower=lower, count=int(selected.sum()),
                predicted=float(p[selected].mean()), actual=float(f[selected].mean())))
    return dict(events=len(x), remaining_minutes_mae=float(residual.abs().mean()),
        remaining_minutes_rmse=float(residual.square().mean().sqrt()),
        explained_variance=float(1 - residual.var(unbiased=False) / y.var(unbiased=False).clamp_min(1e-6)),
        failure_brier=float((p-f).square().mean()), failure_accuracy=float(((p >= .5) == f.bool()).float().mean()),
        failure_fraction=float(f.mean()), predicted_failure=float(p.mean()), calibration=bins)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/convergence_diagnosis_20261005/frozen')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/convergence_diagnosis_20261005/grounded_credit')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--critic-epochs', type=int, default=12)
    parser.add_argument('--actor-updates', type=int, default=96)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.manual_seed(87654)
    source_path = args.source/'checkpoints/latest.pt'
    checkpoint = torch.load(source_path, map_location='cpu', weights_only=False)
    actor_state = checkpoint['training_state']['models']['actor']; del checkpoint
    adapter = make_adapter(args.source, args.output/'before')
    adapter.actor.load_state_dict(actor_state, strict=True)
    original_hash = tree_hash(adapter.actor.state_dict())
    atomic_json(args.output/'protocol.json', dict(source=str(source_path), source_sha256=sha256_file(source_path),
        source_actor_hash=original_hash, demand_seed=314159, indices=list(range(6)), actor_seeds=list(range(271828,271834)),
        critic_inputs='exact actor10 + current observed dynamic flow4 + current time fraction',
        labels='factual per-vehicle remaining observed minutes and deadline failure',
        objective='expected predicted remaining time / 60 + predicted failure probability, KL constrained',
        failure_penalty_minutes=60, kl_mean_max=.03, critic_epochs=args.critic_epochs,
        actor_updates=args.actor_updates, no_world_model=True,
        warning='Objective addition diagnostic. Unexecuted action predictions extrapolate; not exact Q^pi.'))
    if not (args.output/'before/results.json').exists() or len(json.loads((args.output/'before/results.json').read_text())) != 6:
        before = collect(adapter, args.output/'before', range(6), capture=True)
    else:
        before = json.loads((args.output/'before/results.json').read_text())
    arrays = []
    for index in range(6):
        with np.load(args.output/f'before/credit/episode_{index:03d}.npz') as archive:
            arrays.append({k:archive[k] for k in archive.files})
    data = {k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}; del arrays
    features = torch.from_numpy(data['features']); mask = torch.from_numpy(data['mask'])
    action = torch.from_numpy(data['action']); factual = features[torch.arange(len(features)), action]
    y = torch.from_numpy(data['remaining_minutes']); f = torch.from_numpy(data['failure'])
    rng = np.random.default_rng(87654)
    # Every event of a vehicle goes in one split, including repeated route loops.
    vehicle_keys = np.unique(data['vehicle'][data['demand_index'] < 5]); rng.shuffle(vehicle_keys)
    valid_keys = vehicle_keys[:len(vehicle_keys)//5]
    is_val = np.isin(data['vehicle'], valid_keys) & (data['demand_index'] < 5)
    is_train = ~is_val & (data['demand_index'] < 5)
    is_held = data['demand_index'] == 5
    train_ids = np.flatnonzero(is_train); device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device); torch.cuda.set_per_process_memory_fraction(.06, device)
    critic = RealCostCritic().to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    for epoch in range(args.critic_epochs):
        rng.shuffle(train_ids); losses=[]
        for start in range(0, len(train_ids), 4096):
            ids = torch.from_numpy(train_ids[start:start+4096])
            predicted_t, predicted_f = critic(factual[ids].to(device))
            loss = ((predicted_t-y[ids].to(device)/60).square().mean()
                + torch.nn.functional.binary_cross_entropy_with_logits(predicted_f, f[ids].to(device)))
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(critic.parameters(),10); optimizer.step()
            losses.append(float(loss.detach()))
        print(json.dumps(dict(stage='critic', epoch=epoch+1, loss=float(np.mean(losses)))), flush=True)
    stats = {name:critic_metrics(critic, factual[selected], y[selected], f[selected], device)
        for name, selected in [('train', is_train), ('vehicle_validation', is_val), ('heldout_demand5', is_held)]}
    atomic_json(args.output/'critic_metrics.json', stats)
    torch.save(critic.cpu().state_dict(), args.output/'critic.pt'); critic.to(device).requires_grad_(False)
    actor = copy.deepcopy(adapter.actor).to(device)
    reference = copy.deepcopy(actor).requires_grad_(False)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    # Use train states only. One vehicle receives equal aggregate mass regardless
    # of how many loop decisions its source trajectory happened to contain.
    unique, counts = np.unique(data['vehicle'][train_ids], return_counts=True)
    count_map = dict(zip(unique, counts)); weights=np.asarray([1/count_map[v] for v in data['vehicle'][train_ids]])
    weights/=weights.sum()
    chosen = rng.choice(train_ids, min(16384,len(train_ids)), replace=False, p=weights)
    x = features[chosen].to(device); legal=mask[chosen].to(device)
    with torch.no_grad():
        time_prediction, failure_prediction = critic(x)
        cost=(time_prediction+failure_prediction.sigmoid()).masked_fill(~legal,0)
        old_p=reference.forward_logits(x[...,:10],legal).softmax(-1)
    rows=[]
    for update in range(args.actor_updates):
        probability=actor.forward_logits(x[...,:10],legal).softmax(-1)
        kl=(old_p*(old_p.clamp_min(1e-20).log()-probability.clamp_min(1e-20).log())).sum(-1).mean()
        expected=(probability*cost).sum(-1).mean()
        loss=expected+2*kl
        old_state=copy.deepcopy(actor.state_dict())
        actor_optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),1);actor_optimizer.step()
        with torch.no_grad():
            new_p=actor.forward_logits(x[...,:10],legal).softmax(-1)
            new_kl=(old_p*(old_p.clamp_min(1e-20).log()-new_p.clamp_min(1e-20).log())).sum(-1).mean()
        row=dict(update=update+1,expected_cost=float(expected.detach()),kl=float(new_kl),loss=float(loss.detach()))
        rows.append(row)
        if float(new_kl)>.03:
            actor.load_state_dict(old_state);row['reverted_due_to_kl']=True;break
    atomic_json(args.output/'actor_fit.json', rows)
    torch.save(actor.cpu().state_dict(), args.output/'actor.pt'); adapter.actor=actor.cpu()
    after=collect(adapter,args.output/'after',range(6))
    assert [(r['index'],r['demand_hash']) for r in before]==[(r['index'],r['demand_hash']) for r in after]
    result=dict(before=aggregate(before),after=aggregate(after),critic=stats,
        actor_updates_accepted=sum(not r.get('reverted_due_to_kl',False) for r in rows),
        source_actor_hash=original_hash,final_actor_hash=tree_hash(actor.state_dict()),
        limitation='Fitted factual labels; alternate actions extrapolated. Paired demands are reused training demand groups.')
    if result['after']['completion_rate']>result['before']['completion_rate']:
        fresh_before_actor=NextHopActor(); fresh_before_actor.load_state_dict(actor_state); adapter.actor=fresh_before_actor
        fresh_before=collect(adapter,args.output/'heldout_before',range(6,9))
        adapter.actor=actor;fresh_after=collect(adapter,args.output/'heldout_after',range(6,9))
        result['fresh_heldout']=dict(before=aggregate(fresh_before),after=aggregate(fresh_after))
    atomic_json(args.output/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
