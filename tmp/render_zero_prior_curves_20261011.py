"""Render training and verified zero-prior evaluations after the rollout change."""
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

RUN = Path('outputs/ingolstadt_full_retrain_20261009/online_bounded_dynamic_fresh_2000_20261011')
OUT = RUN / 'curves_zero_prior_rollout8_20261011'
OUT.mkdir(exist_ok=True)
raw = (RUN / 'pilot_progress.json').read_bytes()
(OUT / 'pilot_progress_snapshot.json').write_bytes(raw)
progress = json.loads(raw)
initial = json.loads((RUN / 'zero_prior_initial_actor_20261011/summary.json').read_text())
online = sorted(progress['online'], key=lambda row: row['episode_index'])
episodes = np.array([row['episode_index'] + 1 for row in online])
assert all(row['metrics']['complete'] for row in online)
reference_hashes = {r['episode_index']: r['demand_hash'] for r in initial['evaluations']}
fixed = [{
    'online_episodes': 0,
    'mean_observed_minutes_per_started_vehicle': initial['mean_observed_minutes_per_started_vehicle'],
    'weighted_completion_rate': initial['vehicle_weighted_completion_rate'],
    'censored': True,
    'prior_weight': 0,
    'evaluations': initial['evaluations'],
}]
evaluation_rows = [(progress['evaluation_reference_online_episodes'], progress['before'])]
evaluation_rows += [(r['online_episodes'], r['evaluations']) for r in progress['milestones'] if r['phase'] == 'completed']
for count, rows in evaluation_rows:
    assert {r['episode_index']: r['demand_hash'] for r in rows} == reference_hashes
    for row in rows:
        m = row['metrics']
        assert m['evaluation_sp_prior_weight'] == 0
        assert np.isfinite(m['evaluation_global_reward_sum'])
        assert np.isclose(m['evaluation_selection_score'], m['evaluation_global_reward_sum'])
    fixed.append({
        'online_episodes': count,
        'mean_observed_minutes_per_started_vehicle': float(np.mean([r['metrics']['evaluation_t_all'] for r in rows])),
        'weighted_completion_rate': sum(r['arrived_count'] for r in rows) / sum(r['total_vehicle_count'] for r in rows),
        'censored': any(r['travel_time_is_censored'] for r in rows),
        'prior_weight': 0,
        'evaluations': rows,
    })

keys = ['online_t_all', 'online_completion_rate', 'latent_critic_loss', 'event_critic_loss',
        'policy_entropy_mean', 'world_model_loss', 'imagination_seconds', 'online_wall_seconds']
arrays = {key: np.array([r['metrics'][key] for r in online], dtype=float) for key in keys}
assert all(np.isfinite(v).all() for v in arrays.values())
for row in online:
    m = row['metrics']
    assert np.isclose(m['online_selection_score'], m['online_global_reward_sum'])
result = {
    'snapshot_utc': datetime.now(timezone.utc).isoformat(),
    'completed_online_episodes': len(online),
    'rollout_change_after_online_episode': 99,
    'rollouts_before_change': 16, 'rollouts_after_change': 8,
    'evaluation_prior_weight': 0,
    'guided_evaluation_points_used': False,
    'fixed_demand_hashes_match': True,
    'zero_prior_evaluations': fixed,
    'first20_means': {k: float(v[:20].mean()) for k, v in arrays.items()},
    'recent20_means': {k: float(v[-20:].mean()) for k, v in arrays.items()},
    'notes': [
        'The zero-prior evaluation curve contains measured points only; no points are available between 0 and 99.',
        'Episode 0 observed time is censored exposure, not completed travel time.',
        'Online collection uses a dynamic prior; zero-prior evaluation measures the actor without that prior.',
        'Recent 20 averages contain both rollout budgets until 20 episodes have completed after the change.',
        'Critic training loss and bounds do not establish actual-return calibration.',
    ],
}
(OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')

font_names = {font.name for font in font_manager.fontManager.ttflist}
font = next((f for f in ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Serif CJK SC'] if f in font_names), None)
plt.rcParams.update({'font.family': font or 'DejaVu Sans', 'axes.unicode_minus': False, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
def text(en, cn):
    return cn if font else en
def trailing(v):
    return np.array([v[max(0, i-19):i+1].mean() for i in range(len(v))])
def series(ax, key, name, color='#2866ae', scale=1):
    v = arrays[key] * scale
    ax.plot(episodes, v, linewidth=.9, color=color, alpha=.25)
    ax.plot(episodes, trailing(v), linewidth=2, color=color, label=name)
fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
ax = axes[0, 0]
series(ax, 'online_t_all', text('Trailing 20 mean', '最近20轮均值'))
ax.set_title(text('Online observed time (training prior enabled)', '在线观测时间（训练使用动态先验）'))
ax.set_ylabel(text('Minutes / started vehicle', '分钟/已出发车辆'))
ax.legend()
ax = axes[0, 1]
series(ax, 'online_completion_rate', text('Online trailing 20', '在线最近20轮均值'), scale=100)
ax.scatter([r['online_episodes'] for r in fixed], [100*r['weighted_completion_rate'] for r in fixed],
           color='#dc7438', marker='o', zorder=4, label=text('Zero-prior fixed evaluation', '零先验固定评估'))
ax.set_ylim(-2, 104)
ax.set_title(text('Completion: training vs zero-prior evaluation', '到达率：训练采样与零先验评估'))
ax.set_ylabel('%')
ax.annotate('0: 0.13%', (0, initial['vehicle_weighted_completion_rate']*100), xytext=(8, 12), textcoords='offset points')
ax.legend(loc='center left')
ax = axes[0, 2]
counts = [r['online_episodes'] for r in fixed]
times = [r['mean_observed_minutes_per_started_vehicle'] for r in fixed]
ax.scatter(counts, times, color='#2866ae', s=50, zorder=4,
           label=text('Measured zero-prior points only', '仅显示实测零先验评估点'))
ax.set_yscale('log')
ax.set_ylim(3, 70)
ax.set_yticks([3, 4, 10, 20, 50], labels=['3', '4', '10', '20', '50'])
for row in fixed:
    count = row['online_episodes']
    value = row['mean_observed_minutes_per_started_vehicle']
    note = f'{count}: {value:.3f}'
    if row['censored']:
        note += text(' (censored)', '（截断观测）')
    ax.annotate(note, (count, value), xytext=(8, -16 if count == 100 else 12),
                textcoords='offset points', ha='left' if count == 0 else 'right')
ax.set_title(text('Zero-prior fixed evaluation (log axis)', '零先验固定评估（对数刻度）'))
ax.set_ylabel(text('Observed minutes / started vehicle', '观测分钟/已出发车辆'))
ax.set_xlim(-5, max(episodes[-1]+8, 110))
ax.legend(loc='center left')
ax = axes[1, 0]
series(ax, 'latent_critic_loss', 'V loss')
series(ax, 'event_critic_loss', 'Q loss', '#dc7438')
ax.set_yscale('log')
ax.set_title(text('Critic losses (trailing 20)', 'Critic损失（最近20轮均值）'))
ax.set_ylabel(text('MSE, log axis', 'MSE（对数刻度）'))
ax.legend()
ax = axes[1, 1]
series(ax, 'policy_entropy_mean', text('Trailing 20 mean', '最近20轮均值'))
ax.set_title(text('Policy entropy', '策略熵'))
ax.set_ylabel('Nats')
ax.legend()
ax = axes[1, 2]
series(ax, 'world_model_loss', text('Trailing 20 mean', '最近20轮均值'))
ax.set_title(text('World-model training loss', '世界模型训练损失'))
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax.legend()
for ax in axes.flat:
    ax.axvline(99, color='#646464', linestyle='--', linewidth=1, alpha=.8)
    ax.grid(alpha=.18)
    ax.set_xlabel(text('Completed online episodes; dashed: 16→8 at 99', '已完成在线轮数；虚线：99轮后16→8'))
fig.suptitle(text(f'Ingolstadt | {len(online)} episodes | evaluation prior = 0 | rollout 16→8',
                   f'Ingolstadt | 已完成{len(online)}轮 | 评估先验=0 | rollout 16→8'), fontsize=17)
fig.savefig(OUT / 'training_curves.png', dpi=180)
fig.savefig(OUT / 'training_curves.pdf')
print(json.dumps({'count':len(online), 'zero_prior_points': [{k:r[k] for k in ['online_episodes', 'mean_observed_minutes_per_started_vehicle', 'weighted_completion_rate','censored']} for r in fixed], 'output':str(OUT)}, indent=2))
