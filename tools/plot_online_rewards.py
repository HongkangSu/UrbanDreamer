"""Plot real online rewards and same-engine fixed-demand evaluations separately."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    raw = (args.run_dir / 'pilot_progress.json').read_text()
    progress = json.loads(raw)
    reference = int(progress['evaluation_reference_online_episodes'])
    train = []
    for episode in progress['online']:
        if episode['episode_index'] < reference:
            continue  # Earlier episodes used a different SUMO engine.
        metrics = episode['metrics']
        reward = float(metrics['online_global_reward_sum'])
        nonarrivals = int(metrics['online_nonarrival_count'])
        score = float(metrics['online_selection_score'])
        if not np.isclose(score, reward - 5000 * nonarrivals, rtol=0, atol=1e-6):
            raise ValueError('logged selection score violates the recorded reward contract')
        train.append(dict(episode=episode['episode_index'] + 1, shared_reward=reward,
            selection_score=score, completion=metrics['online_completion_rate'],
            nonarrivals=nonarrivals, population=metrics['online_vehicle_count']))
    evals = []
    sources = [(reference, progress['before'])] + [
        (row['online_episodes'], row['evaluations']) for row in progress['milestones']
        if row['phase'] == 'completed']
    expected_hashes = [row['demand_hash'] for row in progress['before']]
    for episode, rows in sources:
        if [row['demand_hash'] for row in rows] != expected_hashes:
            raise ValueError('fixed evaluation demand identities changed')
        if any(row['engine_provenance'] != progress['before'][0]['engine_provenance'] for row in rows):
            raise ValueError('evaluation engine changed')
        population = sum(row['total_vehicle_count'] for row in rows)
        evals.append(dict(episode=episode, shared_reward=sum(
            row['metrics']['evaluation_global_reward_sum'] * row['total_vehicle_count']
            for row in rows) / population,
            selection_score=np.mean([row['metrics']['evaluation_selection_score'] for row in rows]),
            completion=sum(row['arrived_count'] for row in rows) / population,
            nonarrivals=sum(row['nonarrival_count'] for row in rows), population=population))
    for rows in (train, evals):
        if not rows or not np.isfinite([[row[key] for key in row] for row in rows]).all():
            raise ValueError('empty or nonfinite plot data')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'progress_snapshot.json').write_text(raw)
    for name, rows in [('training', train), ('evaluation', evals)]:
        with (args.output_dir / (name + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    blocks = []
    for start, end in [(51,100),(101,200),(201,300),(301,400),(401,train[-1]['episode'])]:
        selected = [row for row in train if start <= row['episode'] <= end]
        if selected:
            blocks.append(dict(start=start, end=end, count=len(selected), **{
                key: float(np.mean([row[key] for row in selected]))
                for key in ('shared_reward','completion','nonarrivals','selection_score')}))
    summary = dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),
        completed_episodes=len(progress['online']), engine=progress['before'][0]['engine_provenance'],
        training_blocks=blocks, latest_evaluation=evals[-1],
        best_evaluation_by_score=max(evals, key=lambda row: row['selection_score']),
        note='Shared reward is negative observed vehicle-minutes/population; censored trajectories are not completed trip-time labels. Selection score additionally sums individual nonarrival penalties and is not the actor loss. Training stochastic, evaluations argmax. Only libsumo episodes included.')
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    names = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ('Noto Sans CJK SC','Noto Sans CJK JP','Noto Serif CJK SC','Noto Serif CJK JP') if name in names), 'DejaVu Sans')
    plt.rcParams.update({'font.family':font, 'axes.unicode_minus':False, 'font.size':11,
        'axes.spines.top':False, 'axes.spines.right':False, 'savefig.dpi':170})
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), layout='constrained', sharex='col')
    fig.get_layout_engine().set(rect=(0, .065, 1, .95))
    specifications = [('shared_reward',1.,'共享交通奖励（负分钟／车）'),
        ('selection_score',1e-6,'含未到达惩罚的排名分（百万）'),
        ('completion',100.,'到达完成率（%）')]
    for column, rows in enumerate((train, evals)):
        x = np.array([row['episode'] for row in rows])
        for row_index, (key, scale, label) in enumerate(specifications):
            axis = axes[row_index, column]
            if column == 1 and key == 'selection_score':
                scale, label = .001, '三组平均排名分（千）'
            y = np.array([row[key] for row in rows]) * scale
            if column == 0:
                axis.plot(x, y, color='#70899b', alpha=.35, lw=.8, label='单轮记录')
                smooth = np.array([np.mean(y[max(0,i-19):i+1]) for i in range(len(y))])
                axis.plot(x, smooth, color='#cc6418', lw=2, label='后向 20 轮均值')
            else:
                axis.plot(x, y, 'o-', color='#187ba0', lw=1.8, ms=4, label='固定三组需求')
                axis.annotate(f'{y[-1]:.2f}', (x[-1],y[-1]), xytext=(-6,9),
                    textcoords='offset points', ha='right', color='#126281')
            axis.axvline(160, color='#777', ls=':', lw=1, alpha=.7)
            axis.grid(alpha=.18); axis.set_ylabel(label)
            if row_index == 0:
                axis.set_title('在线采集：随机策略／每轮新需求' if column == 0 else '固定需求评估：确定性 argmax')
                axis.legend(loc='best', fontsize=9)
            if row_index == 2:
                axis.set_xlabel('已完成在线训练轮次')
                if column == 0: axis.set_ylim(0, 102)
    fig.suptitle(f'Ingolstadt 奖励与完成率｜截至第 {train[-1]["episode"]} 轮', fontsize=17, y=1.02)
    fig.text(.5,.023,'所有曲线越高越好；左右纵轴范围不同。仅包含同一 libsumo 1.27.1 阶段。\n'
        '虚线：第 160 轮后启用并行想象（时间标记，不代表因果归因）。共享奖励含截断观测费用，不能当作完整平均旅行时间。',
        ha='center', fontsize=10, color='#4b5563')
    fig.savefig(args.output_dir / 'reward_curves.png', bbox_inches='tight')
    fig.savefig(args.output_dir / 'reward_curves.pdf', bbox_inches='tight')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
