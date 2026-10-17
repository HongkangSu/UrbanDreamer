"""Plot the completed Ingolstadt run from one immutable progress snapshot."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'outputs/urbandreamer_current_to_dest_1000_fp16_w8_r8_p2_20261013'
RUN = BATCH / 'ingolstadt_custom/online'
NOW = datetime.now(timezone.utc)
OUT = BATCH / ('ingo_final_reward_curves_' + NOW.strftime('%Y%m%d_%H%M%S'))
OUT.mkdir()
raw = (RUN / 'pilot_progress.json').read_text()
(OUT / 'progress_snapshot.json').write_text(raw)
progress = json.loads(raw)
assert progress['phase'] == 'completed' and len(progress['online']) == 1000
train = []
for row in progress['online']:
    m = row['metrics']
    assert np.isclose(m['online_global_reward_sum'], -m['online_t_all'], rtol=0, atol=1e-6)
    train.append(dict(episode=row['episode_index']+1, reward=m['online_global_reward_sum'],
        completion=m['online_completion_rate'], population=m['online_vehicle_count'],
        nonarrival=m['online_nonarrival_count'], seconds=row['seconds'],
        entropy=m['policy_entropy_mean']))
sources = [(0, progress['before'])] + [(m['online_episodes'], m['evaluations'])
    for m in progress['milestones'] if m['phase'] == 'completed'] + [(1000, progress['after'])]
reference = {r['episode_index']: r['demand_hash'] for r in progress['before']}
fixed = []
for episode, rows in sources:
    assert len(rows) == 3 and {r['episode_index']: r['demand_hash'] for r in rows} == reference
    rewards = [r['metrics']['evaluation_global_reward_sum'] for r in rows]
    assert all(np.isclose(r['metrics']['evaluation_global_reward_sum'],
                         -r['metrics']['evaluation_t_all'], rtol=0, atol=1e-6) for r in rows)
    fixed.append(dict(episode=episode, reward=float(np.mean(rewards)),
        reward_min=min(rewards), reward_max=max(rewards),
        completion=sum(r['arrived_count'] for r in rows)/sum(r['total_vehicle_count'] for r in rows),
        nonarrival=sum(r['nonarrival_count'] for r in rows)))
for name, rows in [('training.csv', train), ('evaluation.csv', fixed)]:
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trailing(values):
    return np.array([np.mean(values[max(0, i-19):i+1]) for i in range(len(values))])


blocks = []
for start, end in [(1,20),(101,200),(301,400),(501,600),(701,800),(901,1000),(981,1000)]:
    rows = train[start-1:end]
    blocks.append(dict(start=start, end=end, reward=float(np.mean([r['reward'] for r in rows])),
        completion=sum(r['population']-r['nonarrival'] for r in rows)/sum(r['population'] for r in rows),
        seconds=float(np.mean([r['seconds'] for r in rows]))))
summary = dict(snapshot_utc=NOW.isoformat(), completed_episodes=1000, phase=progress['phase'],
    completed_utc=json.loads((BATCH / 'ingolstadt_custom/status.json').read_text())['updated_utc'],
    training_blocks=blocks, final_evaluation=fixed[-1],
    best_completed_evaluation=max((r for r in fixed if r['completion']==1.), key=lambda r:r['reward']),
    evaluation_points=len(fixed), fixed_demand_hashes_verified=True,
    cache_handoff_boundaries=[490,505],
    note='Sampled training uses changing demands; argmax uses the same three fixed demands. Reward is negative observed minutes per started vehicle. Cache lines mark runtime changes, not causal improvement.')
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
names = {font.name for font in font_manager.fontManager.ttflist}
font = next((n for n in ['Noto Sans CJK SC','Noto Sans CJK JP','Noto Serif CJK SC'] if n in names), 'DejaVu Sans')
plt.rcParams.update({'font.family':font, 'axes.unicode_minus':False, 'font.size':11,
    'axes.spines.top':False, 'axes.spines.right':False, 'savefig.dpi':180})
fig, axes = plt.subplots(2,2,figsize=(13.5,9))
fig.subplots_adjust(top=.79,bottom=.12,left=.075,right=.985,hspace=.30,wspace=.18)
x=np.array([r['episode'] for r in train]); y=np.array([r['reward'] for r in train])
c=np.array([r['completion'] for r in train])*100; entropy=np.array([r['entropy'] for r in train])
fx=np.array([r['episode'] for r in fixed]); fy=np.array([r['reward'] for r in fixed]); fc=np.array([r['completion'] for r in fixed])*100
for ax in axes[0]:
    ax.plot(x,y,color='#4382be',alpha=.22,lw=.8)
    ax.plot(x,trailing(y),color='#1761ad',lw=2.2)
    ax.plot(fx,fy,'o--',color='#d97825',lw=1.7,ms=3.6)
axes[0,0].set_title('完整奖励曲线',pad=10)
axes[0,0].set_ylabel('奖励（负的观测分钟／车）↑ 越高越好')
axes[0,0].set_ylim(-48,1)
axes[0,1].set_title('后半程奖励放大：第 500–1000 ep',pad=10)
axes[0,1].set_xlim(500,1000); axes[0,1].set_ylim(-6,-3.2)
axes[0,1].text(.96,.08,f'最终 argmax：{fy[-1]:.3f}\n最近20 ep采样均值：{trailing(y)[-1]:.3f}',
    transform=axes[0,1].transAxes,ha='right',bbox=dict(facecolor='white',edgecolor='none',alpha=.9),fontsize=10)
axes[1,0].plot(x,c,color='#4382be',alpha=.25,lw=.8)
axes[1,0].plot(x,trailing(c),color='#1761ad',lw=2.2)
axes[1,0].plot(fx,fc,'o--',color='#d97825',lw=1.7,ms=3.6)
axes[1,0].set_title('车辆到达率',pad=10); axes[1,0].set_ylabel('到达率（%）'); axes[1,0].set_ylim(-3,105)
axes[1,1].plot(x,entropy,color='#4382be',alpha=.23,lw=.8)
axes[1,1].plot(x,trailing(entropy),color='#1761ad',lw=2.2)
axes[1,1].axhline(np.log(4),color='#7a7a7a',ls=':',lw=1)
axes[1,1].text(980,np.log(4)-.065,'ln(4)：四槽位均匀策略',ha='right',fontsize=9,color='#666')
axes[1,1].set_title('想象训练中的策略熵',pad=10); axes[1,1].set_ylabel('熵（nats）'); axes[1,1].set_ylim(0,1.5)
for ax in axes.flat:
    ax.grid(alpha=.18); ax.set_xlabel('已完成在线训练 episode')
    if ax is not axes[0,1]: ax.set_xlim(0,1000)
    ax.axvspan(490,505,color='#777',alpha=.09)
fig.suptitle('Ingolstadt：已正常完成 1000 ep',fontsize=20,y=.965)
fig.text(.5,.916,'固定30秒完整路径选路  ·  FP16 imagination  ·  8 workers / 8 rollouts  ·  2 passes / ep',
    ha='center',fontsize=11,color='#4b5563')
fig.legend(handles=[Line2D([0],[0],color='#4382be',alpha=.35,lw=1,label='采样训练：单个 episode'),
    Line2D([0],[0],color='#1761ad',lw=2.2,label='采样训练：后向20 ep均值'),
    Line2D([0],[0],color='#d97825',marker='o',ls='--',lw=1.7,label='固定三组需求：argmax评估')],
    loc='upper center',bbox_to_anchor=(.5,.887),ncol=3,frameon=False,fontsize=10)
fig.text(.5,.035,'灰色区域：第490–505 ep期间切换共享缓存（时间标记，不代表因果归因）。\n'
    '奖励包含未到达车辆的截断观测时间，需结合到达率；策略熵来自想象训练。',
    ha='center',fontsize=9,color='#4b5563')
fig.savefig(OUT / 'ingo_reward_curves.png')
fig.savefig(OUT / 'ingo_reward_curves.pdf')
print(json.dumps(dict(output=str(OUT),**summary),indent=2))
