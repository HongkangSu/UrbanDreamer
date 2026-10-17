"""Plot checkpoint-committed stochastic training rewards for the current batch."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.plot_current_three_city_baselines import read_snapshot, smooth_segments, finite, segments
from tools.queued_city_baselines import durable_episode

BATCH = ROOT / 'outputs/city_baselines_current_to_dest_1000_20261013_180607'
NOW = datetime.now(timezone.utc)
OUT = BATCH / ('sampling_reward_curves_' + NOW.strftime('%Y%m%d_%H%M%S'))
OUT.mkdir()
METHODS = ('online_ac','gnac','dreamer','mbpo','ippo','mappo','iql','mamba','dta')
LABELS = dict(zip(METHODS, ('Online AC','GNAC','Dreamer','MBPO','IPPO','MAPPO','IQL','Mamba','DTA')))
COLORS = dict(zip(METHODS, ('#2463a5','#22825c','#e88b12','#8546b4','#d65a8e',
                          '#a56a27','#21a6ac','#b13037','#333333')))
CITIES = dict(ingolstadt_custom='Ingolstadt', coulommiers='Coulommiers',
              montereau_fault_yonne='Montereau')
manifest_raw = (BATCH / 'manifest.json').read_bytes()
manifest = json.loads(manifest_raw)
ledger = json.loads((BATCH / 'queue_state.json').read_text())
(OUT / 'manifest_snapshot.json').write_bytes(manifest_raw)
(OUT / 'queue_snapshot.json').write_text(json.dumps(ledger, indent=2) + '\n')
data, summaries, exported = {}, [], []
for entry in manifest['runs']:
    city, method = entry['city'], entry['method']
    if city not in CITIES or method not in METHODS:
        continue
    cursor = durable_episode(entry)
    records, quality = read_snapshot(Path(entry['output_dir']) / 'metrics.jsonl')
    rows = [r for r in records if r['phase']=='train' and r['policy']=='stochastic' and r['episode']<=cursor]
    assert rows and len({r['episode'] for r in rows}) == len(rows), (city,method)
    assert all(r['objective']=='observed_mean' for r in rows), (city,method)
    for row in rows:
        assert finite(row['global_reward_sum']) and finite(row['canonical_reward_minutes']), (city,method,row['episode'])
        assert np.isclose(row['global_reward_sum'], row['canonical_reward_minutes'], rtol=0, atol=1e-5), (city,method,row['episode'])
    for key in ('canonical_reward_minutes','completion_rate'):
        for row, value in zip(rows, smooth_segments(rows,key,50)):
            row[key+'_ma50'] = value
    state = ledger['jobs'][entry['id']]
    label = '已完成' if state['status']=='completed' else '运行中' if state['status']=='running' else '停止恢复' if state.get('user_requested_skip_recovery') else state['status']
    recent, first = rows[-50:], rows[:50]
    item = dict(city=city, method=method, label=LABELS[method], status=state['status'], status_label=label,
        checkpoint_episode=cursor, plotted_through_episode=rows[-1]['episode'], training_rows=len(rows),
        target_episodes=entry['episode_budget'], first50_reward=float(np.mean([r['canonical_reward_minutes'] for r in first])),
        recent50_reward=float(np.mean([r['canonical_reward_minutes'] for r in recent])),
        recent50_completion=float(np.mean([r['completion_rate'] for r in recent])),
        data_quality=quality)
    summaries.append(item)
    data[(city,method)] = (rows,item)
    exported.extend(dict(city=city,method=method,**row) for row in rows)
    print(json.dumps(dict(city=city,method=method,episode=rows[-1]['episode'],recent50_reward=item['recent50_reward'])),flush=True)
assert len(data)==27
with (OUT/'sampling_rewards.csv').open('w',newline='') as stream:
    fields=list(exported[0])
    writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader(); writer.writerows(exported)
summary=dict(snapshot_started_utc=NOW.isoformat(), snapshot_finished_utc=datetime.now(timezone.utc).isoformat(),
    smoothing_window=50, episode_budget=2000, reward='logged global_reward_sum, verified equal to -observed_mean_seconds/60',
    source='phase=train, policy=stochastic, at or before each checkpoint cursor; last replay wins; incomplete final lines ignored',
    note='These are real training collection rewards, not fixed-demand evaluations. DTA is its training assignment/collection reward. Missing future episodes are not extrapolated.', runs=summaries)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
names={font.name for font in font_manager.fontManager.ttflist}
font=next((n for n in ('Noto Sans CJK SC','Noto Sans CJK JP','Noto Serif CJK SC') if n in names),'DejaVu Sans')
plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})


def plot_run(ax, rows, color, *, metric='canonical_reward_minutes', scale=1., raw=True):
    for chunk in segments(rows):
        x=np.array([r['episode'] for r in chunk])
        y=np.array([r[metric] for r in chunk],dtype=float)*scale
        smooth=np.array([r[metric+'_ma50'] for r in chunk],dtype=float)*scale
        if raw: ax.plot(x,y,color=color,alpha=.16,lw=.6)
        ax.plot(x,smooth,color=color,lw=1.9)
    ax.scatter(rows[-1]['episode'],rows[-1][metric+'_ma50']*scale,color=color,s=19,zorder=5)


fig,axes=plt.subplots(1,3,figsize=(16,6.2),sharey=True)
fig.subplots_adjust(top=.73,bottom=.19,left=.055,right=.985,wspace=.11)
for ax,(city,name) in zip(axes,CITIES.items()):
    for method in METHODS:
        rows,item=data[(city,method)]
        plot_run(ax,rows,COLORS[method])
    ax.set_title(name,fontsize=14,pad=12)
    ax.set_xlim(0,2025); ax.set_ylim(-65,.7); ax.set_xlabel('已保存的训练 episode')
    ax.grid(alpha=.18)
axes[0].set_ylabel('采样训练奖励（负的观测分钟／车）↑ 越高越好')
fig.suptitle('三个城市各 baseline：采样训练奖励曲线',fontsize=19,y=.965)
fig.text(.5,.905,'浅线：单个 episode  ·  粗线：后向50 ep均值  ·  圆点：该任务当前末端',ha='center',fontsize=11,color='#4b5563')
handles=[Line2D([0],[0],color=COLORS[m],lw=2.2,label=LABELS[m]) for m in METHODS]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.865),ncol=9,frameon=False,fontsize=10)
fig.text(.5,.075,f'快照：{NOW.strftime("%Y-%m-%d %H:%M:%S UTC")}。目标均为2000 ep；曲线止于已保存进度，未外推。\n'
    '奖励包含未到达车辆的截断观测时间；结合到达率解读。Montereau MAPPO保留至742 ep，按既定记录停止恢复。',
    ha='center',fontsize=9,color='#4b5563')
for extension in ('png','pdf'): fig.savefig(OUT/f'all_baselines_sampling_reward.{extension}')
plt.close(fig)
for city,name in CITIES.items():
    fig,axes=plt.subplots(3,3,figsize=(15,10.7))
    fig.subplots_adjust(top=.86,bottom=.11,left=.065,right=.98,hspace=.52,wspace=.24)
    for ax,method in zip(axes.flat,METHODS):
        rows,item=data[(city,method)]
        plot_run(ax,rows,COLORS[method])
        y=np.array([r['canonical_reward_minutes'] for r in rows])
        pad=max(.15,(float(y.max())-float(y.min()))*.08)
        ax.set_ylim(float(y.min())-pad,float(y.max())+pad)
        ax.set_xlim(0,max(20,rows[-1]['episode'])*1.025)
        ax.set_title(f'{LABELS[method]} | {rows[-1]["episode"]}/2000 | {item["status_label"]}',fontsize=11,color=COLORS[method],pad=8)
        ax.set_xlabel('训练 episode',fontsize=9); ax.set_ylabel('奖励 ↑',fontsize=9)
        ax.grid(alpha=.18); ax.tick_params(labelsize=9)
        ax.text(.96,.06,f'MA50: {item["recent50_reward"]:.2f}\n到达率: {item["recent50_completion"]*100:.1f}%',
            transform=ax.transAxes,ha='right',va='bottom',fontsize=9,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
    fig.suptitle(f'{name}：各 baseline 采样训练奖励',fontsize=18,y=.975)
    fig.text(.5,.925,'浅线：单个 episode；粗线：后向50 ep均值。各子图横、纵轴自适应，请按数值比较。',
        ha='center',fontsize=11,color='#4b5563')
    fig.text(.5,.035,f'快照：{NOW.strftime("%Y-%m-%d %H:%M:%S UTC")}；只含检查点已保存的真实训练采集。\n'
        '标注到达率为最近50 ep均值。DTA曲线为其训练分配／采集回报。未到达车辆的时间为截止时的观测值。',
        ha='center',fontsize=9,color='#4b5563')
    for extension in ('png','pdf'): fig.savefig(OUT/f'{city}_sampling_reward_details.{extension}')
    plt.close(fig)
print(json.dumps(dict(output=str(OUT),runs=len(summaries),training_points=len(exported),snapshot_finished_utc=summary['snapshot_finished_utc']),ensure_ascii=False),flush=True)
