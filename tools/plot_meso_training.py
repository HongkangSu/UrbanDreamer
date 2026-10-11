"""Freeze a live mainline progress file and plot traffic outcomes and learning separately."""
from __future__ import annotations

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


METRICS = ('world_model_loss', 'active_fraction_loss', 'actor_loss',
    'latent_critic_loss', 'event_critic_loss', 'policy_entropy_mean',
    'actor_advantage_mean', 'actor_advantage_std', 'counterfactual_baseline_mean',
    'counterfactual_baseline_std')


def trailing(values, width=20):
    return np.asarray([np.mean(values[max(0, i-width+1):i+1]) for i in range(len(values))])


def build_rows(progress):
    training = []
    for record in progress['online']:
        m = record['metrics']
        reward, failures = float(m['online_global_reward_sum']), int(m['online_nonarrival_count'])
        if not np.isclose(m['online_selection_score'], reward - 5000*failures, atol=1e-6, rtol=0):
            raise ValueError('selection score differs from the reward contract')
        training.append(dict(episode=record['episode_index']+1, shared_reward=reward,
            completion=float(m['online_completion_rate']), nonarrivals=failures,
            population=int(m['online_vehicle_count']), seconds=float(record['seconds']),
            **{name: float(m[name]) for name in METRICS}))
    reference = int(progress.get('evaluation_reference_online_episodes', 0))
    sources = [(reference, progress['before'])] + [(entry['online_episodes'], entry['evaluations'])
        for entry in progress['milestones'] if entry['phase'] == 'completed']
    expected = [(r['demand_hash'], r['total_vehicle_count'], r['engine_provenance'])
                for r in progress['before']]
    evaluations = []
    for episode, rows in sources:
        if [(r['demand_hash'], r['total_vehicle_count'], r['engine_provenance']) for r in rows] != expected:
            raise ValueError('fixed demand, population or engine changed')
        population = sum(r['total_vehicle_count'] for r in rows)
        if population <= 0:
            raise ValueError('evaluation has no vehicles')
        reward = sum(r['metrics']['evaluation_global_reward_sum']*r['total_vehicle_count']
                     for r in rows) / population
        failures = sum(r['nonarrival_count'] for r in rows)
        evaluations.append(dict(episode=episode, shared_reward=reward,
            completion=1-failures/population, nonarrivals=failures, population=population,
            observed_minutes=-reward, completed_travel_minutes=-reward if failures == 0 else None))
    if not training or not evaluations:
        raise ValueError('completed training and evaluation rows are required')
    if not np.isfinite([[r[k] for k in training[0]] for r in training]).all():
        raise ValueError('nonfinite training metric')
    return training, evaluations


def draw_trace(axis, x, y, label, *, log=False):
    y = np.asarray(y)
    if log:
        y = np.maximum(y, 1e-12)
        axis.set_yscale('log')
    axis.plot(x, y, color='#8298a9', alpha=.35, lw=.8, label='单轮')
    axis.plot(x, trailing(y), color='#c86120', lw=2, label='后向 20 轮均值')
    axis.set_ylabel(label)
    axis.grid(alpha=.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--progress-snapshot', type=Path,
        help='Reuse a frozen progress JSON for reproducible rendering')
    args = parser.parse_args()
    raw = (args.progress_snapshot or args.run_dir/'pilot_progress.json').read_text()
    progress = json.loads(raw)
    train, evaluation = build_rows(progress)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output/'progress_snapshot.json').write_text(raw)
    for name, rows in [('training', train), ('evaluation', evaluation)]:
        with (output/f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    def block(first, last):
        rows = [r for r in train if first <= r['episode'] <= last]
        n = sum(r['population'] for r in rows)
        return dict(start=first, end=last, episodes=len(rows), population=n,
            completion=1-sum(r['nonarrivals'] for r in rows)/n,
            shared_reward=sum(r['shared_reward']*r['population'] for r in rows)/n,
            **{k: float(np.mean([r[k] for r in rows])) for k in (*METRICS, 'seconds')})
    count = len(train)
    blocks = [block(first, min(first+49, count)) for first in range(1, count+1, 50)]
    summary = dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),
        run_dir=str(args.run_dir.resolve()), completed_episodes=count, target_episodes=2000,
        phase=progress['phase'], first20=block(1, min(20,count)),
        last20=block(max(1,count-19),count), training_blocks=blocks,
        evaluations=evaluation, latest_training=train[-1],
        latest_evaluation=evaluation[-1],
        best_complete_evaluation=min((r for r in evaluation if r['nonarrivals'] == 0),
            key=lambda r:r['completed_travel_minutes'], default=None),
        note='Training uses stochastic actions and new demand each episode. Evaluations use argmax and the same three demands. Shared reward is negative observed minutes per vehicle; censored episodes are not completed travel-time labels. Actor loss is a signed policy surrogate, not a traffic reward.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    names = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ('Noto Sans CJK SC','Noto Sans CJK JP','Noto Serif CJK SC')
                 if name in names), 'DejaVu Sans')
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
    x = np.asarray([r['episode'] for r in train])
    ex = [r['episode'] for r in evaluation]
    fig, axes = plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
    fig.get_layout_engine().set(rect=(0,.09,1,.80))
    draw_trace(axes[0,0],x,[r['shared_reward'] for r in train],'共享交通奖励（负观测分钟／车）')
    draw_trace(axes[1,0],x,[100*r['completion'] for r in train],'完成率（%）')
    for ax,key,scale,label in ((axes[0,1],'shared_reward',1.,'共享交通奖励（负观测分钟／车）'),
                               (axes[1,1],'completion',100.,'完成率（%）')):
        ys = [scale*r[key] for r in evaluation]
        ax.plot(ex,ys,'o-',color='#167897',lw=2,ms=4,label='固定三组需求，按车辆数加权')
        ax.annotate(f'{ys[-1]:.2f}',(ex[-1],ys[-1]),xytext=(-8,-16 if key=='completion' else 9),
            textcoords='offset points',ha='right',color='#12647f')
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0,0].set_title('真实在线采集：随机策略，每轮新需求')
    axes[0,1].set_title('真实固定需求评估：确定性 argmax')
    for ax in axes[1]:
        ax.set_ylim(0,104)
    for ax in axes.flat:
        ax.set_xlabel('已完成在线训练轮次')
    axes[0,0].legend(fontsize=9)
    axes[0,1].legend(fontsize=9)
    fig.suptitle(f'Ingolstadt 中观主线｜真实奖励与完成率｜截至 {count} / 2000 轮',fontsize=16,y=.98)
    fig.text(.5,.035,'奖励与完成率越高越好。奖励曲线包含截止时仍未到达车辆的已观测费用；\n'
        '只有全员到达的评估点，其负奖励才能解释为完整平均旅行时间。所有评估采用同一需求、同一中观环境。',
        ha='center',fontsize=10,color='#475569')
    for extension in ('png','pdf'):
        fig.savefig(output/f'real_outcomes.{extension}')
    plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(13,11),layout='constrained')
    fig.get_layout_engine().set(rect=(0,.07,1,.83))
    specs=[('world_model_loss','世界模型 FM loss',True),
        ('event_critic_loss','事件 Q critic loss',True),
        ('latent_critic_loss','宏观 V critic loss',True),
        ('actor_loss','Actor loss（带符号）',False),
        ('policy_entropy_mean','策略熵',False)]
    for ax,(key,label,log) in zip(axes.flat,specs):
        draw_trace(ax,x,[r[key] for r in train],label,log=log)
        ax.set_title(label)
    ax=axes[2,1]
    mean=trailing([r['actor_advantage_mean'] for r in train])
    deviation=trailing([r['actor_advantage_std'] for r in train])
    ax.plot(x,mean,color='#c86120',lw=2,label='20 轮优势均值')
    ax.fill_between(x,mean-deviation,mean+deviation,color='#c86120',alpha=.15,label='± 轮内标准差的 20 轮均值')
    ax.axhline(0,color='#64748b',lw=1,ls='--')
    ax.set_title('Actor 优势')
    ax.set_ylabel('优势')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    for ax in axes.flat:
        ax.set_xlabel('已完成在线训练轮次')
    axes[0,0].legend(fontsize=9)
    fig.suptitle(f'Ingolstadt 中观主线｜学习指标｜截至第 {count} 轮',fontsize=16,y=.985)
    fig.text(.5,.023,'前三张 loss 使用对数纵轴；浅线为每轮记录，深线为后向 20 轮均值。\n'
        'Actor loss 可正可负，不要求单调下降；loss 下降不等于真实交通性能或世界模型预测误差已收敛。',
        ha='center',fontsize=10,color='#475569')
    for extension in ('png','pdf'):
        fig.savefig(output/f'learning_curves.{extension}')
    plt.close(fig)
    print(json.dumps({k:summary[k] for k in ('snapshot_utc','completed_episodes','first20','last20',
        'latest_evaluation','best_complete_evaluation')},indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
