"""Snapshot committed five-baseline metrics and plot training/evaluation separately."""
import csv
from datetime import datetime, timezone
import hashlib
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
from tools.plot_current_three_city_baselines import complete_evaluations, normalize, smooth_segments, segments
from tools.queued_city_baselines import durable_episode

BATCH = ROOT / 'outputs/city_baselines_static_round2_2000_20261015_173045'
NOW = datetime.now(timezone.utc)
OUT = BATCH / ('curves_' + NOW.strftime('%Y%m%d_%H%M%S'))
OUT.mkdir()
METHODS = ('ippo', 'mappo', 'mamba', 'mbpo', 'dreamer')
LABELS = dict(zip(METHODS, ('IPPO', 'MAPPO', 'MAMBA', 'MBPO', 'Dreamer')))
COLORS = dict(zip(METHODS, ('#d24c47', '#2468b4', '#2b9564', '#8a54b6', '#e69522')))
CITIES = {'ingolstadt_custom': 'Ingolstadt', 'coulommiers': 'Coulommiers',
          'montereau_fault_yonne': 'Montereau'}
WINDOW = 20
manifest = json.loads((BATCH / 'manifest.json').read_text())
status = json.loads((BATCH / 'status.json').read_text())
(OUT / 'manifest_snapshot.json').write_text(json.dumps(manifest, indent=2) + '\n')
(OUT / 'status_snapshot.json').write_text(json.dumps(status, indent=2) + '\n')
eval_seeds = [manifest['seed'] + 10000 + i for i in range(manifest['eval_seeds'])]
runs, summaries, exported_train, exported_eval = {}, [], [], []

for entry in manifest['runs']:
    city, method = entry['city'], entry['method']
    if city not in CITIES or method not in METHODS:
        continue
    cursor = durable_episode(entry)
    path = Path(entry['output_dir']) / 'metrics.jsonl'
    raw = path.read_bytes()
    (OUT / f'{city}_{method}_metrics.jsonl').write_bytes(raw)
    unique, duplicates, partial = {}, 0, 0
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b'\n'):
            partial += 1
            continue
        row = json.loads(line)
        key = tuple(row[k] for k in ('phase', 'episode', 'policy', 'seed'))
        duplicates += key in unique
        value = normalize(row)
        value['branch_entropy'] = row.get('branch_entropy')
        unique[key] = value
    records = sorted(unique.values(), key=lambda r: (r['episode'], r['phase'], r['policy'], r['seed']))
    records = [r for r in records if r['episode'] <= cursor]
    train = [r for r in records if r['phase'] == 'train' and r['policy'] == 'stochastic']
    assert train and len({r['episode'] for r in train}) == len(train)
    assert all(r['objective'] == 'observed_mean' and
        np.isclose(r['global_reward_sum'], r['canonical_reward_minutes'], atol=1e-5, rtol=0)
        for r in train)
    for metric in ('canonical_reward_minutes', 'completion_rate', 'branch_entropy'):
        assert all(np.isfinite(r[metric]) for r in train)
        for row, value in zip(train, smooth_segments(train, metric, WINDOW)):
            row[metric + '_ma20'] = value
    evaluations, incomplete = complete_evaluations(records, eval_seeds)
    runs[(city, method)] = dict(train=train, evaluation=evaluations)
    summaries.append(dict(city=city, method=method, checkpoint_episode=cursor,
        train_points=len(train), latest_episode=train[-1]['episode'],
        last20_reward=float(np.mean([r['canonical_reward_minutes'] for r in train[-20:]])),
        last20_completion=float(np.mean([r['completion_rate'] for r in train[-20:]])),
        last20_entropy=float(np.mean([r['branch_entropy'] for r in train[-20:]])),
        evaluation_points=len(evaluations), incomplete_evaluations=incomplete,
        duplicate_rows_replaced=duplicates, partial_lines_ignored=partial,
        metrics_sha256=hashlib.sha256(raw).hexdigest()))
    exported_train.extend(dict(city=city, method=method, **r) for r in train)
    exported_eval.extend(dict(city=city, method=method, **r) for r in evaluations)
assert len(runs) == 15
for name, rows in (('training.csv', exported_train), ('evaluation.csv', exported_eval)):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

fonts = {f.name for f in font_manager.fontManager.ttflist}
font = next((f for f in ('Noto Sans CJK SC', 'Noto Sans CJK JP') if f in fonts), 'DejaVu Sans')
plt.rcParams.update({'font.family': font, 'axes.unicode_minus': False, 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 170})
handles = [Line2D([0], [0], color=COLORS[m], lw=2.3, label=LABELS[m]) for m in METHODS]


def train_line(ax, rows, method, metric, scale=1.):
    for chunk in segments(rows):
        x = [r['episode'] for r in chunk]
        ax.plot(x, [r[metric] * scale for r in chunk], color=COLORS[method], alpha=.13, lw=.65)
        ax.plot(x, [r[metric + '_ma20'] * scale for r in chunk], color=COLORS[method], lw=1.9)
    ax.scatter(rows[-1]['episode'], rows[-1][metric + '_ma20'] * scale,
               color=COLORS[method], s=27, zorder=5)


def decorate(fig, axes, title, subtitle):
    compact = fig.get_figheight() < 7
    fig.suptitle(title, fontsize=20, y=.985)
    fig.text(.5, .91 if compact else .935, subtitle, ha='center', fontsize=10, color='#4b5563')
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .87 if compact else .91),
               ncol=5, frameon=False, fontsize=11)
    for ax in np.asarray(axes).flat:
        ax.grid(alpha=.18)
        ax.set_xlabel('已保存的训练集数（episode）')
    fig.text(.5, .025, f'快照 {NOW:%Y-%m-%d %H:%M UTC}  |  仅本轮重启实验  |  目标 2000 集；曲线在实际进度处结束，不外推\n'
        '各城市横轴范围不同；当前队列因磁盘保护暂停。奖励包含未到达车辆的截止观测时间，需结合完成率解读。',
        ha='center', fontsize=9, color='#4b5563')


def save(fig, name):
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{extension}')
    plt.close(fig)


fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), sharey=True)
fig.subplots_adjust(left=.065, right=.985, top=.75, bottom=.20, wspace=.10)
for ax, (city, title) in zip(axes, CITIES.items()):
    for method in METHODS:
        train_line(ax, runs[(city, method)]['train'], method, 'canonical_reward_minutes')
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-55, 1)
    ax.set_xlim(0, max(runs[(city, m)]['train'][-1]['episode'] for m in METHODS) * 1.035)
axes[0].set_ylabel('采样训练回报（负的观测分钟/车）↑ 越高越好')
decorate(fig, axes, '五个 baseline：当前采样训练回报', '浅线：每集原始值   ·   粗线：后向 20 集均值（不足20集用已有样本）   ·   圆点：当前末端')
save(fig, 'training_rewards')

fig, axes = plt.subplots(2, 3, figsize=(17, 9.4), sharey='row', sharex='col')
fig.subplots_adjust(left=.065, right=.985, top=.81, bottom=.14, wspace=.10, hspace=.28)
for col, (city, title) in enumerate(CITIES.items()):
    for method in METHODS:
        rows = runs[(city, method)]['train']
        train_line(axes[0, col], rows, method, 'completion_rate', 100.)
        train_line(axes[1, col], rows, method, 'branch_entropy')
    axes[0, col].set_title(title, fontsize=14)
    axes[0, col].set_ylim(0, 103)
    axes[1, col].set_ylim(0, 1.46)
    axes[1, col].axhline(np.log(4), color='#555555', ls=':', lw=1)
    axes[1, col].text(.98, .98, '四槽均匀分布熵 ln(4)', transform=axes[1, col].transAxes,
        ha='right', va='top', fontsize=9, color='#555555')
    axes[1, col].set_xlim(0, max(runs[(city, m)]['train'][-1]['episode'] for m in METHODS) * 1.035)
axes[0, 0].set_ylabel('采样训练完成率（%）↑')
axes[1, 0].set_ylabel('采样策略熵（nats）')
decorate(fig, axes, '完成率与策略熵：识别失败和近均匀策略', '浅线：每集原始值   ·   粗线：后向20集均值   ·   熵下降本身不代表策略变好')
save(fig, 'training_completion_entropy')

fig, axes = plt.subplots(2, 3, figsize=(17, 9.4), sharey='row', sharex='col')
fig.subplots_adjust(left=.065, right=.985, top=.81, bottom=.14, wspace=.10, hspace=.28)
for col, (city, title) in enumerate(CITIES.items()):
    missing = []
    for method in METHODS:
        rows = runs[(city, method)]['evaluation']
        if not rows:
            missing.append(LABELS[method])
            continue
        for chunk in segments(rows, max_gap=100):
            for ax, metric, scale in ((axes[0, col], 'canonical_reward_minutes', 1.),
                                      (axes[1, col], 'completion_rate', 100.)):
                ax.errorbar([r['episode'] for r in chunk], [r[metric] * scale for r in chunk],
                    yerr=[r[metric + '_std'] * scale for r in chunk], color=COLORS[method],
                    marker='D', markersize=4, lw=1.7, capsize=3)
    axes[0, col].set_title(title, fontsize=14)
    axes[0, col].set_ylim(-55, 1)
    axes[1, col].set_ylim(0, 103)
    axes[1, col].set_xlim(0, max(runs[(city, m)]['train'][-1]['episode'] for m in METHODS) * 1.035)
    if missing:
        axes[0, col].text(.03, .05, '尚无完整评估：\n' + '、'.join(missing), transform=axes[0, col].transAxes,
            fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
axes[0, 0].set_ylabel('确定性评估回报 ↑ 越高越好')
axes[1, 0].set_ylabel('确定性评估完成率（%）↑')
decorate(fig, axes, '单独看 argmax 评估：不与采样训练混画',
    '每100集评估一次；仅绘制3个固定需求种子齐全的结果。点：均值；误差棒：需求种子标准差，不是置信区间。')
save(fig, 'deterministic_evaluation')

summary = dict(snapshot_utc=NOW.isoformat(), batch=str(BATCH), smoothing_window=WINDOW,
    train_points=len(exported_train), evaluation_groups=len(exported_eval),
    reward_definition='global_reward_sum = -observed_mean_seconds / 60',
    source='latest replay per phase/episode/policy/seed, bounded by checkpoint cursor',
    status_counts=status['counts'], runs=summaries)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(dict(output=str(OUT), train_points=len(exported_train),
                     evaluation_groups=len(exported_eval)), indent=2))
