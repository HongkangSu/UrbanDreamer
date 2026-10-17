"""Read-only failed MAPPO probability reconstruction; no simulator or optimizer."""
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'outputs/city_baselines_current_to_dest_1000_20261013_180607'
SOURCE = BATCH / 'source'
sys.path.insert(0, str(SOURCE))
import numpy as np
import torch
from torch.distributions import Categorical
from urban_dreamer.learning.baselines.common import accounting_context
from urban_dreamer.learning.observations import build_actor_observations, prepare_actor_window
from urban_dreamer.models.policy import NextHopActor, set_shortest_path_prior
from urban_dreamer.traffic.route_candidates import RouteCandidates
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
checkpoint_path = BATCH / 'montereau_fault_yonne/mappo/latest.pt'
checkpoint_sha = sha(checkpoint_path)
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
episode = checkpoint['episode'] + 1
archive_path = checkpoint_path.parent / f'episodes/train/episode_{episode:06d}.npz'
with np.load(archive_path, allow_pickle=False) as data:
    summary = json.loads(data['summary_json'].tobytes())
    states = json.loads(data['states_json'].tobytes())
    events = json.loads(data['policy_decisions_json'].tobytes())
    flow = data['flow'].copy()
config = load_config(checkpoint['args']['config'])
topology = load_traffic_topology(validate_prepared_inputs(config))
candidates = RouteCandidates(topology.network, traffic_model='meso')
deadline = summary['deadline_seconds']
actor = NextHopActor().eval()
actor.load_state_dict(checkpoint['method_state']['actor'], strict=True)
actor.score_gauge = checkpoint['method_state']['learning_protocol']['score_gauge']
set_shortest_path_prior(actor, 0., 30.)
by_window = defaultdict(list)
for event in events:
    by_window[event['window_index']].append(event)
features, masks, actions, ordered_events, batch_ranges = [], [], [], [], []
cpu_collection_logits, cpu_collection_logprob = [], []
duplicate_actions = 0
started = time.monotonic()
for index, window_events in sorted(by_window.items()):
    state = states[index]
    vehicles = {vehicle['vehicle_id']: vehicle for vehicle in state['vehicles']}
    ordered = sorted(window_events, key=lambda event: event['vehicle_id'])
    pending = [vehicles[event['vehicle_id']] for event in ordered]
    assert all(vehicle['current_edge'] == event['current_edge']
        and float(event['decision_time']) == state['time']
        for vehicle, event in zip(pending, ordered))
    context = accounting_context(state, deadline)
    observation = build_actor_observations(
        policy_edge_features=prepare_actor_window(flow[index][:, [2, 1]], topology.network.edge_count),
        vehicle_states=pending, candidates=candidates,
        episode_progress=context[0], active_vehicle_fraction=context[1])
    chosen = []
    for row, event in enumerate(ordered):
        matches = [slot for slot, route in enumerate(observation.routes[row]) if route == tuple(event['route'])]
        assert matches
        duplicate_actions += len(matches) > 1
        assert all(np.array_equal(observation.features[row, matches[0]], observation.features[row, slot])
                   for slot in matches)
        chosen.append(matches[0])
    begin = sum(len(value) for value in actions)
    features.append(observation.features.copy())
    masks.append(observation.mask.copy())
    actions.append(np.asarray(chosen))
    ordered_events.extend(ordered)
    batch_ranges.append((begin, begin + len(ordered)))
    with torch.no_grad():
        logits = actor.forward_logits(torch.from_numpy(features[-1]), torch.from_numpy(masks[-1]))
        cpu_collection_logits.append(logits)
        cpu_collection_logprob.append(Categorical(logits=logits).log_prob(torch.tensor(chosen)))
    if index % 20 == 0:
        print(json.dumps(dict(stage='reconstruct', episode=episode, window=index,
            rows=len(ordered_events), seconds=time.monotonic() - started)), flush=True)
features = torch.from_numpy(np.concatenate(features))
mask = torch.from_numpy(np.concatenate(masks))
actions = torch.from_numpy(np.concatenate(actions))
reference_logits = torch.cat(cpu_collection_logits)
reference_logprob = torch.cat(cpu_collection_logprob)
reference_probs = Categorical(logits=reference_logits).probs


@torch.no_grad()
def infer(model, device, ranges, dtype=torch.float32):
    logits = torch.cat([model.forward_logits(features[begin:end].to(device=device, dtype=dtype),
        mask[begin:end].to(device)).cpu() for begin, end in ranges])
    distribution = Categorical(logits=logits)
    return logits, distribution.log_prob(actions), distribution.probs


chunk_ranges = [(begin, min(begin + 512, len(features))) for begin in range(0, len(features), 512)]
variants = {'cpu512': infer(actor, 'cpu', chunk_ranges)}
double_actor = copy.deepcopy(actor).double()
variants['cpu64_512'] = infer(double_actor, 'cpu', chunk_ranges, torch.float64)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(.02, 0)
    gpu_actor = copy.deepcopy(actor).to('cuda:0')
    variants['cuda512'] = infer(gpu_actor, 'cuda:0', chunk_ranges)
    variants['cuda_collection_batches'] = infer(gpu_actor, 'cuda:0', batch_ranges)
report = dict(schema_version=1, checkpoint_episode=checkpoint['episode'], failed_episode=episode,
    checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
    checkpoint_unchanged=sha(checkpoint_path) == checkpoint_sha,
    archive=str(archive_path), archive_sha256=sha(archive_path), policy_events=len(features),
    collection_batches=len(batch_ranges), collection_batch_sizes=[end-begin for begin,end in batch_ranges],
    archive_collection_device=summary['metadata'].get('collection_device'),
    saved_cli_collection_device=checkpoint['args'].get('collection_device'),
    runtime=json.loads((checkpoint_path.parent / 'collection_runtime.json').read_text()),
    score_gauge=actor.score_gauge, prior_weight=actor.prior_weight,
    duplicate_selected_route_slots=duplicate_actions, comparisons={},
    source_sha256={name: sha(SOURCE/name) for name in (
        'urban_dreamer/learning/baselines/ppo.py', 'urban_dreamer/models/policy.py',
        'urban_dreamer/learning/baselines/collection_policy.py')},
    reconstruction_note='Actual failed full-route archive supplies current edge, remaining distance, frozen flow, causal context and one sorted batch per global boundary. Selected duplicate slots have identical features. Collected old_log_prob is not archived, so this reproduces it by CPU inference with the unchanged preceding checkpoint rather than reading original in-memory values.',
    scope='No SUMO, sampling, backward pass, optimizer step, checkpoint write, process change or frozen-source edit.')
for name, (logits, logprob, probs) in variants.items():
    error = (logprob - reference_logprob).abs()
    threshold = 1e-4 + 1e-5 * reference_logprob.abs()
    failures = error > threshold
    top = []
    for item in error.topk(min(8, len(error))).indices:
        i = int(item)
        event = ordered_events[i]
        top.append(dict(index=i, window=event['window_index'], vehicle=event['vehicle_id'],
            selected_slot=int(actions[i]), cpu_collection_logprob=float(reference_logprob[i]),
            other_logprob=float(logprob[i]), absolute_error=float(error[i]), threshold=float(threshold[i]),
            cpu_logits=reference_logits[i][mask[i]].tolist(), other_logits=logits[i][mask[i]].tolist(),
            features=features[i].tolist(), mask=mask[i].tolist()))
    report['comparisons'][name] = dict(max_logprob_absolute_error=float(error.max()),
        guard_failure_count=int(failures.sum()), max_probability_absolute_error=float((probs-reference_probs).abs().max()),
        max_legal_logit_absolute_error=float((logits[mask]-reference_logits[mask]).abs().max()), top=top)
report['reconstructed_cross_device_guard_failure'] = report['comparisons'].get('cuda512', {}).get('guard_failure_count', 0) > 0
path = BATCH / 'mappo_montereau_behavior_failure_20261014.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({name: {key:value for key,value in comparison.items() if key != 'top'}
    for name,comparison in report['comparisons'].items()}, indent=2), flush=True)
