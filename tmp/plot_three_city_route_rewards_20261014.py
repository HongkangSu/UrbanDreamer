"""Snapshot current route runs and plot their recorded real traffic rewards."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/urbandreamer_current_to_dest_1000_fp16_w8_r8_p2_20261013'
NOW = datetime.now(timezone.utc)
OUT = RUN / ('reward_curves_' + NOW.strftime('%Y%m%d_%H%M%S'))
OUT.mkdir()
SPECS = [('ingolstadt_custom', 'Ingolstadt'), ('coulommiers', 'Coulommiers'),
         ('montereau_fault_yonne', 'Montereau')]


def trailing(values, window=20):
    return np.array([np.mean(values[max(0, i-window+1):i+1]) for i in range(len(values))])


def export(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


names = {font.name for font in font_manager.fontManager.ttflist}
font = next((name for name in ('Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Serif CJK SC')
             if name in names), 'DejaVu Sans')
plt.rcParams.update({'font.family': font, 'axes.unicode_minus': False, 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 180})
fig, axes = plt.subplots(2, 3, figsize=(15, 8.7), sharey='row')
fig.subplots_adjust(top=.78, bottom=.14, left=.065, right=.985, hspace=.25, wspace=.12)
summary = dict(snapshot_utc=NOW.isoformat(), smoothing='trailing mean over up to 20 episodes',
    reward='logged global traffic reward = -observed minutes per started vehicle',
    evaluation_aggregation='arithmetic mean of three fixed-demand episode rewards', cities=[])
for column, (city, label) in enumerate(SPECS):
    runtime_status = json.loads((RUN / city / 'status.json').read_text())
    budget = json.loads((RUN / city / 'online/effective_run_budget.json').read_text())
    target = budget['authoritative_runner_online_episodes']
    state_label = ('已完成' if runtime_status['status'] == 'completed' else
                   'OOM停止' if runtime_status['status'] == 'failed' else '运行中')
    raw = (RUN / city / 'online/pilot_progress.json').read_text()
    (OUT / (city + '_progress_snapshot.json')).write_text(raw)
    progress = json.loads(raw)
    train = []
    for row in progress['online']:
        m = row['metrics']
        assert m['complete']
        assert np.isclose(m['online_global_reward_sum'], -m['online_t_all'], rtol=0, atol=1e-6)
        assert np.isclose(m['online_selection_score'], m['online_global_reward_sum'], rtol=0, atol=1e-6)
        train.append(dict(episode=row['episode_index'] + 1, reward=m['online_global_reward_sum'],
                          completion=m['online_completion_rate'], vehicle_count=m['online_vehicle_count'],
                          nonarrival_count=m['online_nonarrival_count'], seconds=row['seconds']))
    reference = {row['episode_index']: row['demand_hash'] for row in progress['before']}
    sources = [(0, progress['before'])] + [(row['online_episodes'], row['evaluations'])
        for row in progress.get('milestones', []) if row['phase'] == 'completed']
    if progress.get('phase') == 'completed' and progress.get('after'):
        sources.append((len(train), progress['after']))
    fixed = []
    for count, rows in sources:
        assert len(rows) == 3 and {r['episode_index']: r['demand_hash'] for r in rows} == reference
        rewards = []
        for row in rows:
            m = row['metrics']
            assert m['evaluation_sp_prior_weight'] == 0
            assert np.isclose(m['evaluation_global_reward_sum'], -m['evaluation_t_all'], rtol=0, atol=1e-6)
            rewards.append(m['evaluation_global_reward_sum'])
        population = sum(r['total_vehicle_count'] for r in rows)
        fixed.append(dict(episode=count, reward=float(np.mean(rewards)),
            reward_min=min(rewards), reward_max=max(rewards),
            completion=sum(r['arrived_count'] for r in rows) / population,
            vehicle_count=population, nonarrival_count=sum(r['nonarrival_count'] for r in rows)))
    assert train and fixed
    assert np.isfinite([[row['reward'], row['completion']] for row in train + fixed]).all()
    export(city + '_training.csv', train)
    export(city + '_evaluation.csv', fixed)
    x = np.array([r['episode'] for r in train])
    reward = np.array([r['reward'] for r in train])
    completion = np.array([r['completion'] for r in train]) * 100
    fx = np.array([r['episode'] for r in fixed])
    fy = np.array([r['reward'] for r in fixed])
    fc = np.array([r['completion'] for r in fixed]) * 100
    for ax, values, smooth, ev in ((axes[0, column], reward, trailing(reward), fy),
                                  (axes[1, column], completion, trailing(completion), fc)):
        ax.plot(x, values, color='#3875ba', alpha=.24, lw=.8)
        ax.plot(x, smooth, color='#1761ad', lw=2.2)
        ax.plot(fx, ev, 'o--', color='#d97825', ms=4.5, lw=1.8)
        ax.grid(alpha=.18)
        ax.set_xlim(-max(1, len(train)*.015), len(train)*1.025)
    axes[0, column].set_title(f'{label} | {len(train)}/{target} ep | {state_label}', fontsize=12, pad=12)
    axes[0, column].set_ylim(-65, 1)
    axes[1, column].set_ylim(-3, 105)
    axes[1, column].set_xlabel('已完成在线训练 episode')
    axes[0, column].text(.96, .045,
        f'采样 MA20: {trailing(reward)[-1]:.2f}\nargmax @ {fixed[-1]["episode"]}: {fy[-1]:.2f}',
        transform=axes[0, column].transAxes, ha='right', va='bottom', fontsize=9,
        bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
    axes[1, column].text(.96, .38,
        f'采样 MA20: {trailing(completion)[-1]:.1f}%\nargmax @ {fixed[-1]["episode"]}: {fc[-1]:.1f}%',
        transform=axes[1, column].transAxes, ha='right', va='bottom', fontsize=9,
        bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
    recent = train[-20:]
    first = train[:20]
    summary['cities'].append(dict(city=city, completed_episodes=len(train),
        target_episodes=target, runtime_status=runtime_status['status'],
        first20_reward=float(np.mean([r['reward'] for r in first])),
        recent20_reward=float(np.mean([r['reward'] for r in recent])),
        recent20_mean_completion=float(np.mean([r['completion'] for r in recent])),
        recent20_weighted_completion=sum(r['vehicle_count']-r['nonarrival_count'] for r in recent)
                                    / sum(r['vehicle_count'] for r in recent),
        initial_evaluation=fixed[0], latest_evaluation=fixed[-1],
        completed_evaluation_points=len(fixed),
        omitted_incomplete_milestones=[r['online_episodes'] for r in progress.get('milestones', [])
                                      if r['phase'] != 'completed']))
axes[0, 0].set_ylabel('交通奖励（负的观测分钟／车）↑ 越高越好')
axes[1, 0].set_ylabel('车辆到达率（%）↑ 越高越好')
fig.suptitle('UrbanDreamer 主线：奖励曲线与到达率', fontsize=19, y=.965)
fig.text(.5, .913, '30秒完整路径选路  ·  FP16 imagination  ·  8 workers × 8 rollouts  ·  2 passes / ep',
         ha='center', fontsize=11, color='#4b5563')
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([0], [0], color='#3875ba', alpha=.35, lw=1, label='采样训练：单个 episode'),
    Line2D([0], [0], color='#1761ad', lw=2.2, label='采样训练：后向20 ep均值'),
    Line2D([0], [0], color='#d97825', marker='o', ls='--', lw=1.8,
           label='固定需求评估：argmax，完整3组需求')], loc='upper center',
    bbox_to_anchor=(.5, .89), ncol=3, frameon=False, fontsize=10)
fig.text(.5, .055,
    f'快照：{NOW.strftime("%Y-%m-%d %H:%M:%S UTC")}。各城市横轴进度不同，纵轴范围相同。\n'
    '训练为随机采样且每轮需求变化；评估使用固定需求。奖励包含未到达车辆的截断观测时间，需结合到达率解读。',
    ha='center', fontsize=9, color='#4b5563')
fig.savefig(OUT / 'three_city_reward_curves.png')
fig.savefig(OUT / 'three_city_reward_curves.pdf')
(OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(output=str(OUT), **summary), ensure_ascii=False, indent=2))
