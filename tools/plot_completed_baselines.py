"""Plot finished baseline training and fixed-seed evaluation without mixing them."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.plot_current_three_city_baselines import read_snapshot, complete_evaluations, smooth_segments, segments
from tools.compact_episode_archives import committed_episode

METHODS = ('online_ac', 'gnac', 'iql', 'dta')
LABELS = {'online_ac': 'Online AC', 'gnac': 'GNAC', 'iql': 'IQL', 'dta': 'DTA'}
COLORS = {'online_ac': '#2878b5', 'gnac': '#2c9b72', 'iql': '#da6751', 'dta': '#8c60b5'}
CITIES = {'ingolstadt_custom': 'Ingolstadt', 'coulommiers': 'Coulommiers', 'montereau_fault_yonne': 'Montereau'}


def main():
    source = ROOT / 'outputs/city_baselines_current_to_dest_1000_20261013_180607'
    output = ROOT / 'outputs/completed_baseline_curves_20261017'
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / 'manifest.json').read_text())
    seeds = [manifest['seed'] + 10000 + i for i in range(manifest['eval_seeds'])]
    entries = [r for r in manifest['runs'] if r['method'] in METHODS]
    assert len(entries) == 12
    def read(entry):
        records, quality = read_snapshot(Path(entry['output_dir']) / 'metrics.jsonl')
        cursor = committed_episode(entry)
        records = [r for r in records if r['episode'] <= cursor]
        train = [r for r in records if r['phase'] == 'train']
        assert cursor == 2000 and len(train) == 2000 and {r['episode'] for r in train} == set(range(1, 2001))
        assert len({r['episode'] for r in train}) == len(train)
        for metric in ('canonical_reward_minutes', 'completion_rate'):
            assert all(r[metric] is not None and np.isfinite(r[metric]) for r in train)
            for row, value in zip(train, smooth_segments(train, metric, 50)):
                row[metric + '_smoothed'] = value
        evaluation, incomplete = complete_evaluations(records, seeds)
        assert evaluation and evaluation[-1]['episode'] == 2000
        def average(rows, metric):
            return float(np.mean([r[metric] for r in rows]))
        summary = dict(city=entry['city'], method=entry['method'], checkpoint_episode=cursor,
            train_points=len(train), evaluation_points=len(evaluation), incomplete_evaluations=incomplete,
            objectives=sorted({r['objective'] for r in records}), policies=sorted({r['policy'] for r in train}),
            first100_cost_minutes=-average(train[:100], 'canonical_reward_minutes'),
            last100_cost_minutes=-average(train[-100:], 'canonical_reward_minutes'),
            previous100_cost_minutes=-average(train[-200:-100], 'canonical_reward_minutes'),
            first100_completion=average(train[:100], 'completion_rate'),
            last100_completion=average(train[-100:], 'completion_rate'),
            best_eval_episode=max(evaluation, key=lambda r:r['canonical_reward_minutes'])['episode'],
            best_eval_cost_minutes=-max(r['canonical_reward_minutes'] for r in evaluation),
            final_eval_cost_minutes=-evaluation[-1]['canonical_reward_minutes'],
            final_eval_completion=evaluation[-1]['completion_rate'],
            final_eval_cost_std=evaluation[-1]['canonical_reward_minutes_std'], source=quality)
        return entry['city'], entry['method'], train, evaluation, summary
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(read, entries))
    runs = {(city, method): {'train': train, 'evaluation': evaluation}
            for city, method, train, evaluation, _ in results}
    summaries = [summary for *_, summary in results]
    (output / 'summary.json').write_text(json.dumps(dict(methods=METHODS, seeds=seeds,
        smoothing_window=50, results=summaries, uncertainty='SD across three evaluation demand seeds, not independent training seeds'), indent=2)+'\n')
    for phase in ('train', 'evaluation'):
        rows = [dict(city=city, method=method, **row) for city, method, *_ in results for row in runs[(city, method)][phase]]
        fields = sorted(set().union(*(r.keys() for r in rows)))
        with (output / (phase + '.csv')).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 160, 'axes.titleweight': 'bold'})
    handles = [Line2D([0], [0], color=COLORS[m], lw=2.3, label=LABELS[m]) for m in METHODS]
    for phase in ('train', 'evaluation'):
        fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True)
        fig.subplots_adjust(top=.80, bottom=.13, hspace=.25, wspace=.22)
        for col, (city, title) in enumerate(CITIES.items()):
            for method in METHODS:
                rows = runs[(city, method)][phase]
                for k, metric in enumerate(('canonical_reward_minutes', 'completion_rate')):
                    ax = axes[k, col];scale = 100 if k else 1
                    if phase == 'train':
                        for chunk in segments(rows):
                            x = [r['episode'] for r in chunk]
                            ax.plot(x, [r[metric]*scale for r in chunk], color=COLORS[method], alpha=.10, lw=.45)
                            ax.plot(x, [r[metric+'_smoothed']*scale for r in chunk], color=COLORS[method], lw=1.7)
                    else:
                        x = np.array([r['episode'] for r in rows]);y = np.array([r[metric]*scale for r in rows])
                        sd = np.array([r[metric+'_std']*scale for r in rows])
                        ax.plot(x,y,color=COLORS[method],lw=1.6,marker='o',ms=2.7)
                        ax.fill_between(x,y-sd,y+sd,color=COLORS[method],alpha=.10)
            axes[0,col].set_title(title)
            axes[1,col].set_ylim(-2,102)
            for ax in axes[:,col]:
                ax.set_xlim(0,2000);ax.grid(alpha=.18)
            axes[1,col].set_xlabel('Training episode' if phase=='train' else 'Evaluated checkpoint episode')
        axes[0,0].set_ylabel('Observed traffic reward (min/vehicle)\nHigher / closer to zero is better')
        axes[1,0].set_ylabel('Trip completion (%)')
        title = 'Completed baselines: training curves' if phase=='train' else 'Completed baselines: deterministic evaluation'
        subtitle = 'Faint: raw episodes | Solid: trailing 50-episode mean' if phase=='train' else 'Identical 3 demand seeds | Lines: mean | Shading: +/- 1 SD'
        fig.suptitle(title,fontsize=20,y=.97);fig.text(.5,.92,subtitle,ha='center',color='#555555')
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.89),ncol=4,frameon=False)
        fig.text(.5,.035,'Reward = -observed travel time / 60; unfinished trips are censored at the evaluation horizon.\nOne training seed per method/city; evaluation bands are not training-run confidence intervals.',ha='center',fontsize=9,color='#555555')
        for ext in ('png','pdf'):fig.savefig(output/f'{phase}_curves.{ext}')
        plt.close(fig)
    # Separate scales expose learning inside strong methods rather than hiding
    # it behind the large cost of failed policies in the comparison plot.
    fig, axes = plt.subplots(4,3,figsize=(17,12),sharex=True)
    fig.subplots_adjust(top=.90,bottom=.09,hspace=.32,wspace=.23)
    for i,method in enumerate(METHODS):
        for j,(city,title) in enumerate(CITIES.items()):
            ax=axes[i,j];rows=runs[(city,method)]['train']
            for chunk in segments(rows):
                x=[r['episode'] for r in chunk]
                ax.plot(x,[-r['canonical_reward_minutes'] for r in chunk],color=COLORS[method],alpha=.13,lw=.45)
                ax.plot(x,[-r['canonical_reward_minutes_smoothed'] for r in chunk],color=COLORS[method],lw=1.8)
            ax.grid(alpha=.18);ax.set_xlim(0,2000)
            if i==0:ax.set_title(title)
            if j==0:ax.set_ylabel(LABELS[method]+'\nObserved cost (min/vehicle)')
            if i==3:ax.set_xlabel('Training episode')
    fig.suptitle('Within-method learning trajectories (individual y-scales; lower is better)',fontsize=18,y=.96)
    fig.text(.5,.025,'Trailing 50-episode mean; faint raw observations. Compare trends within panels, not visual heights across panels.',ha='center',fontsize=10)
    for ext in ('png','pdf'):fig.savefig(output/f'individual_cost_curves.{ext}')
    plt.close(fig)
    print(json.dumps({'output':str(output),'runs':len(results),'train_rows':sum(len(r[2]) for r in results)},indent=2))


if __name__ == '__main__':
    main()
