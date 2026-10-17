"""Reconstruct one actual AC update from its pre-update checkpoint and archive."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'outputs/city_baselines_current_to_dest_1000_20261013_180607'
sys.path.insert(0, str(BATCH / 'source'))
import numpy as np
import torch
from torch.distributions import Categorical
from tools.train_toy_actor_critic import RealActorCritic, set_episode_entropy
from tools.queued_city_baselines import write_json
from urban_dreamer.learning.episode_archive import read_episode_archive
from urban_dreamer.learning.baselines.common import accounting_context
from urban_dreamer.learning.observations import build_actor_observations, prepare_actor_window
from urban_dreamer.models.policy import individual_actor_advantage
from argparse import Namespace

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
revision = json.loads((BATCH / 'cache_restart_200000.json').read_text())
checkpoint_path = Path(revision['backup']) / 'coulommiers_online_ac/latest.pt'
state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
episode = state['episode'] + 1
args = Namespace(**state['args'])
args.config = Path(args.config)
args.device = 'cpu'
runner = RealActorCritic(args)
runner.actor.load_state_dict(state['actor'])
runner.global_critic.load_state_dict(state['global_critic'])
set_episode_entropy(runner, episode)
archive_path = BATCH / 'coulommiers/online_ac/episodes/train' / f'episode_{episode:06d}.npz'
data = read_episode_archive(archive_path, expected_session_protocol='event')
runner.n = data['metadata']['total_vehicle_count']
states = [{**s, 'flow': data['flow'][i]} for i, s in enumerate(data['states'])]
windows = []
previous_mean = 0.
for index, s in enumerate(states[:-1]):
    next_s = states[index + 1]
    mean = next_s['cumulative_vehicle_seconds'] / max(next_s['started_vehicle_count'], 1)
    windows.append(dict(obs=runner.global_observation(s), next_obs=runner.global_observation(next_s),
        reward=-(mean-previous_mean)/60., terminal=bool(next_s['done'] or next_s['time'] >= runner.deadline)))
    previous_mean = mean
assert abs(sum(w['reward'] for w in windows)-data['metrics']['global_reward_sum']) < 1e-8
global_advantage, targets, values = runner.global_targets(windows)
by_window = defaultdict(list)
for event in data['policy_decisions']:
    by_window[event['window_index']].append(event)
features, masks, actions, event_windows = [], [], [], []
start = time.monotonic()
for index, events in sorted(by_window.items()):
    s = states[index]
    vehicles = {v['vehicle_id']: v for v in s['vehicles']}
    ordered = sorted(events, key=lambda e: e['vehicle_id'])
    pending = [vehicles[e['vehicle_id']] for e in ordered]
    assert all(v['current_edge'] == e['current_edge'] and float(e['decision_time']) == float(s['time'])
               for v, e in zip(pending, ordered))
    context = accounting_context(s, runner.deadline)
    obs = build_actor_observations(policy_edge_features=prepare_actor_window(s['flow'][:, [2, 1]], runner.network.edge_count),
        vehicle_states=pending, candidates=runner.candidates,
        episode_progress=context[0], active_vehicle_fraction=context[1])
    for row, event in enumerate(ordered):
        action = obs.routes[row].index(tuple(event['route']))
        features.append(obs.features[row])
        masks.append(obs.mask[row])
        actions.append(action)
        event_windows.append(index)
    if index % 20 == 0:
        print(json.dumps(dict(episode=episode, reconstructed_through_window=index,
            rows=len(features), elapsed_seconds=time.monotonic()-start)), flush=True)
event_adv = global_advantage[torch.tensor(event_windows)].detach()
advantage = individual_actor_advantage(event_adv, torch.zeros_like(event_adv))
features = torch.from_numpy(np.stack(features))
mask = torch.from_numpy(np.stack(masks))
actions = torch.tensor(actions)
logits = runner.actor.forward_logits(features, mask)
dist = Categorical(logits=logits)
pg_loss = -(dist.log_prob(actions)*advantage).mean()
entropy_loss = -runner.entropy_coefficient*dist.entropy().mean()
parameters = tuple(runner.actor.parameters())
def gradient(loss, retain=True):
    return torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, parameters, retain_graph=retain)])
pg = gradient(pg_loss)
entropy = gradient(entropy_loss)
combined = pg + entropy
cosine = float(torch.dot(pg, entropy)/(pg.norm()*entropy.norm()).clamp_min(1e-20))
record = next(json.loads(line) for line in (BATCH/'coulommiers/online_ac/metrics.jsonl').read_text().splitlines()
              if json.loads(line).get('phase') == 'train' and json.loads(line)['episode'] == episode)
loss_value = float((pg_loss+entropy_loss).detach())
assert abs(loss_value-record['actor_loss']) < 2e-5, (loss_value, record['actor_loss'])
assert abs(float(combined.norm())-record['actor_gradient_norm']) < 2e-6, (float(combined.norm()), record['actor_gradient_norm'])
report = dict(episode=episode, pre_update_episode=state['episode'], checkpoint=str(checkpoint_path),
    archive=str(archive_path), full_actor_rows=len(actions), macro_windows=len(windows),
    exact_logged_loss_reproduced=True, exact_logged_gradient_norm_reproduced=True,
    actor_loss=loss_value, logged_actor_loss=record['actor_loss'],
    policy_gradient_norm=float(pg.norm()), entropy_gradient_norm=float(entropy.norm()),
    combined_gradient_norm=float(combined.norm()), logged_combined_gradient_norm=record['actor_gradient_norm'],
    entropy_to_policy_gradient_norm_ratio=float(entropy.norm()/pg.norm()),
    gradient_cosine=cosine, entropy_coefficient=runner.entropy_coefficient,
    event_advantage_mean=float(event_adv.mean()), event_advantage_std=float(event_adv.std(unbiased=False)),
    actor_advantage_clipped_fraction=float((advantage.abs()>=2.5).float().mean()),
    max_action_probability=float(dist.probs.detach().max(-1).values.mean()),
    normalized_entropy=float((dist.entropy().detach()/mask.sum(-1).float().log()).mean()),
    note='Full reconstruction uses checkpoint immediately before this actual collected update, exact causal active/started context and all archived decisions. Duplicate padded slots have identical features/logits/parameter gradients, so first matching route is equivalent. No optimizer step or SUMO rollout executed.')
write_json(BATCH/'coul_actor_gradient_audit_20261014.json', report)
print(json.dumps(report), flush=True)
