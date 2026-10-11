"""Render a frozen snapshot of the bounded dynamic-prior online experiment."""
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

RUN = Path('outputs/ingolstadt_full_retrain_20261009/online_bounded_dynamic_fresh_2000_20261011')
OUT = RUN / 'curves_before_rollout8_20261011'
OUT.mkdir(exist_ok=True)
payload = (RUN / 'pilot_progress.json').read_bytes()
(OUT / 'pilot_progress_snapshot.json').write_bytes(payload)
data = json.loads(payload)
online = sorted(data['online'], key=lambda row: row['episode_index'])
assert all(row['metrics']['complete'] for row in online)
episode = np.array([row['episode_index'] + 1 for row in online])

def values(key):
    result = np.array([row['metrics'][key] for row in online], dtype=float)
    assert np.all(np.isfinite(result)), key
    return result

for row in online:
    m = row['metrics']
    assert np.isfinite(m['online_global_reward_sum'])
    assert np.isclose(m['online_selection_score'], m['online_global_reward_sum'])
    assert m['online_nonarrival_penalty_sum'] == 0

reference = {row['episode_index']: row['demand_hash'] for row in data['before']}
fixed = []
for count, rows in [(0, data['before'])] + [
    (row['online_episodes'], row['evaluations']) for row in data['milestones']
    if row['phase'] == 'completed'
]:
    assert {row['episode_index']: row['demand_hash'] for row in rows} == reference
    for row in rows:
        m = row['metrics']
        assert np.isfinite(m['evaluation_global_reward_sum'])
        assert np.isclose(m['evaluation_selection_score'], m['evaluation_global_reward_sum'])
        assert m['evaluation_nonarrival_penalty_sum'] == 0
    fixed.append({
        'episodes': count,
        'mean_minutes': float(np.mean([row['observed_minutes_per_vehicle'] for row in rows])),
        'mean_completion': float(np.mean([row['metrics']['evaluation_completion_rate'] for row in rows])),
        'demand_minutes': [row['observed_minutes_per_vehicle'] for row in rows],
        'demand_hashes': [row['demand_hash'] for row in rows],
    })

keys = ['online_t_all', 'online_completion_rate', 'latent_critic_loss', 'event_critic_loss',
        'policy_entropy_mean', 'world_model_loss', 'imagination_seconds', 'online_wall_seconds']
arrays = {key: values(key) for key in keys}
summary = {
    'snapshot_utc': datetime.now(timezone.utc).isoformat(),
    'completed_online_episodes': len(online),
    'first_episode': int(episode[0]), 'last_episode': int(episode[-1]),
    'initial20_means': {k: float(v[:20].mean()) for k, v in arrays.items()},
    'recent20_means': {k: float(v[-20:].mean()) for k, v in arrays.items()},
    'completion_minimum': float(arrays['online_completion_rate'].min()),
    'fixed_evaluations': fixed,
    'fixed_demands_match': True,
    'selection_score_equals_global_reward': True,
    'bound_violations': {
        k: {'sum_logged_pass_means': float(values(k).sum()), 'max_logged_pass_mean': float(values(k).max())}
        for k in ['critic_bound_violations', 'critic_target_bound_violations', 'critic_nonfinite_predictions']
    },
    'notes': ['Online demand changes every episode; moving averages are descriptive, not paired comparisons.',
              'Fixed evaluations use identical three demands and dynamic prior weight 1.',
              'Critic loss and bounds do not establish true-return calibration.',
              'Actor loss is an optimization loss, not a reward.'],
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

available = {font.name for font in font_manager.fontManager.ttflist}
font = next((name for name in ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Serif CJK SC'] if name in available), None)
plt.rcParams.update({'font.family': font or 'DejaVu Sans', 'axes.unicode_minus': False,
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
zh = bool(font)
def label(en, cn):
    return cn if zh else en

fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
colors = ['#2866ae', '#dc7438']
def smooth(v, width=20):
    return np.array([v[max(0, i-width+1):i+1].mean() for i in range(len(v))])
def series(ax, key, name, color=colors[0], scale=1):
    v = arrays[key] * scale
    ax.plot(episode, v, color=color, alpha=.25, linewidth=.9)
    ax.plot(episode, smooth(v), color=color, linewidth=2, label=name)

ax = axes[0, 0]
series(ax, 'online_t_all', label('Trailing 20 mean', '最近20轮均值'))
ax.set_title(label('Online observed travel time (lower is better)', '在线采样平均观测时间（越低越好）'))
ax.set_ylabel(label('Minutes / vehicle', '分钟/车'))
ax.legend()
ax = axes[0, 1]
series(ax, 'online_completion_rate', label('Online trailing 20 mean', '在线最近20轮均值'), scale=100)
ax.plot([r['episodes'] for r in fixed], [r['mean_completion']*100 for r in fixed], 'o--', color=colors[1], label=label('Fixed evaluation', '固定评估'))
ax.set_ylim(95, 100.5) if arrays['online_completion_rate'].min() >= .95 else ax.set_ylim(0, 100.5)
ax.set_ylabel('%')
ax.set_title(label('Completion rate', '车辆到达率'))
ax.legend()
ax = axes[0, 2]
for i in range(3):
    ax.plot([r['episodes'] for r in fixed], [r['demand_minutes'][i] for r in fixed], 'o-', alpha=.35, linewidth=1)
ax.plot([r['episodes'] for r in fixed], [r['mean_minutes'] for r in fixed], 'o-', color=colors[0], linewidth=2.5,
        label=label('Three-demand mean', '三组固定需求均值'))
ax.set_title(label('Paired fixed evaluation (lower is better)', '同需求固定评估（越低越好）'))
ax.set_ylabel(label('Minutes / vehicle', '分钟/车'))
ax.legend()
ax = axes[1, 0]
series(ax, 'latent_critic_loss', 'V loss')
series(ax, 'event_critic_loss', 'Q loss', colors[1])
ax.set_yscale('log')
ax.set_title(label('Critic training losses (trailing 20)', 'Critic训练损失（最近20轮均值）'))
ax.set_ylabel(label('MSE, logarithmic scale', 'MSE（对数刻度）'))
ax.legend()
ax = axes[1, 1]
series(ax, 'policy_entropy_mean', label('Trailing 20 mean', '最近20轮均值'))
ax.set_title(label('Policy entropy', '策略熵'))
ax.set_ylabel(label('Nats', 'Nats'))
ax.legend()
ax = axes[1, 2]
series(ax, 'world_model_loss', label('Trailing 20 mean', '最近20轮均值'))
ax.set_title(label('World-model training loss', '世界模型训练损失'))
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax.legend()
for ax in axes.flat:
    ax.grid(alpha=.18)
    ax.set_xlabel(label('Completed online episodes', '已完成在线训练轮数'))
fig.suptitle(label(f'Ingolstadt bounded critics + dynamic prior | {len(online)} episodes | rollout = 16',
                   f'Ingolstadt 有界critic + 动态先验 | 已完成{len(online)}轮 | rollout = 16'), fontsize=17)
fig.savefig(OUT / 'training_curves.png', dpi=180)
fig.savefig(OUT / 'training_curves.pdf')
print(json.dumps(summary, indent=2))
print('PNG:', OUT / 'training_curves.png')
