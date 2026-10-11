"""Plot sampled completed pretraining losses without implying validation quality."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-log',type=Path,required=True)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    summary=json.loads(args.summary.read_text())
    rows=[]
    for line in args.run_log.read_text().splitlines():
        try: row=json.loads(line)
        except ValueError:continue
        if row.get('phase')=='fm_pretrain':rows.append(row)
    if not rows:raise ValueError('no sampled FM training losses')
    offset=int(summary.get('migration_source_world_model_updates',0))
    epochs=int(summary['pretrain_epochs_completed'])
    updates=int(summary['world_model_initialization_updates'])
    x=np.array([int(row['optimizer_updates'])-offset for row in rows])
    world=np.array([row['world_model_loss'] for row in rows],dtype=float)
    exposure=np.array([row['active_fraction_loss'] for row in rows],dtype=float)
    decoder_available = all('decoder_pretrain_codec_loss' in row for row in rows)
    decoder = np.array([row['decoder_pretrain_codec_loss'] for row in rows], dtype=float) if decoder_available else None
    if not np.isfinite(np.concatenate([world,exposure] + ([decoder] if decoder_available else []))).all():raise ValueError('nonfinite training losses')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'loss_samples.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['utc','epoch','pretrain_update','world_model_loss','exposure_readout_mse','decoder_loss'])
        writer.writerows((r['utc'],int(r['epoch'])+1,int(r['optimizer_updates'])-offset,
                         r['world_model_loss'],r['active_fraction_loss'],r.get('decoder_pretrain_codec_loss')) for r in rows)
    def smooth(values):
        # Sampled training batches differ by traffic state; retain the raw line.
        return np.array([np.mean(values[max(0,i-20):i+1]) for i in range(len(values))])
    available={font.name for font in font_manager.fontManager.ttflist}
    chinese_font = next((name for name in ('Noto Sans CJK SC','Noto Sans CJK JP','Noto Serif CJK SC','Noto Serif CJK JP') if name in available),None)
    chinese=chinese_font is not None
    plt.rcParams.update({'font.family':chinese_font or 'DejaVu Sans',
                         'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.dpi':180})
    panels = 3 if decoder_available else 2
    fig,axes=plt.subplots(panels,1,figsize=(12,3.1*panels+1),sharex=True,layout='constrained')
    fig.get_layout_engine().set(rect=(0,.04,1,.96))
    labels=['世界模型 loss（FM + 0.1 × prior）','在网车辆比例读出 MSE'] if chinese else ['World-model loss (FM + 0.1 × prior)','Active-vehicle fraction readout MSE']
    colors=['#1765aa','#c77713']
    values_by_panel = [world, exposure]
    if decoder_available:
        labels.append('Decoder 监督 loss' if chinese else 'Decoder supervised loss')
        colors.append('#16866b')
        values_by_panel.append(decoder)
    finals=[summary['world_model_loss_last'],summary['active_fraction_loss_last']]
    if decoder_available:
        finals.append(summary['decoder_pretrain_codec_loss_last'])
    for panel,(axis,values,label,color,final) in enumerate(zip(axes,values_by_panel,labels,colors,finals)):
        axis.plot(x/1000,np.maximum(values,1e-12),color=color,alpha=.23,linewidth=.8,
                  label='原始采样批次' if chinese else 'Sampled batch losses')
        axis.plot(x/1000,np.maximum(smooth(values),1e-12),color=color,linewidth=2.1,
                  label='21 点移动平均' if chinese else 'Trailing 21-point mean')
        axis.scatter([updates/1000],[final],marker='*',s=115,color=color,zorder=5,
                     label='最后一次更新' if chinese else 'Final optimizer update')
        if panel < 2:
            axis.set_yscale('log')
        axis.set_ylabel(label);axis.grid(which='major',alpha=.20)
        for epoch in range(1,epochs):axis.axvline(updates/epochs*epoch/1000,color='#7d8692',linestyle='--',linewidth=.8,alpha=.7)
        axis.legend(loc='upper right',frameon=False,ncol=3,fontsize=9)
    axes[-1].set_xlabel('本次预训练参数更新次数（千次）' if chinese else 'Pretraining optimizer updates (thousands)')
    axes[-1].set_xlim(0,updates/1000*1.012)
    top=axes[0].secondary_xaxis('top',functions=(lambda value:value*1000/(updates/epochs),lambda epoch:epoch*(updates/epochs)/1000))
    top.set_xticks(np.arange(epochs+1));top.set_xlabel('完整 epoch 进度' if chinese else 'Completed epoch progress')
    fig.suptitle(f'Ingolstadt 预训练 loss · {epochs} epoch · {int(summary["pretrain_samples_seen"]):,} 个窗口' if chinese else f'Ingolstadt pretraining losses · {epochs} epoch · {int(summary["pretrain_samples_seen"]):,} windows',fontsize=15)
    fig.text(.5,.015,'日志每约 30 秒采样；曲线是训练批次 loss，不能替代独立验证集预测误差。' if chinese else 'Logs sample roughly every 30 seconds. Training losses do not establish held-out prediction quality.',ha='center',fontsize=9,color='#555d66')
    fig.savefig(args.output_dir/'pretrain_loss_curve.png',bbox_inches='tight')
    fig.savefig(args.output_dir/'pretrain_loss_curve.pdf',bbox_inches='tight')
    plt.close(fig)
    stats=[]
    for epoch in range(epochs):
        selected=[row for row in rows if row['epoch']==epoch]
        stats.append(dict(epoch=epoch+1,sampled_points=len(selected),
            world_loss_median=statistics.median(row['world_model_loss'] for row in selected),
            exposure_loss_median=statistics.median(row['active_fraction_loss'] for row in selected)))
    result=dict(sampled_points=len(rows),smoothing='trailing21point_mean',warmup_updates_excluded=offset,
        epochs=stats,final_world_loss=finals[0],final_exposure_loss=finals[1],
        first_world_loss=world[0])
    if epochs > 1:
        result['first_to_last_epoch_world_median_reduction'] = 1-stats[-1]['world_loss_median']/stats[0]['world_loss_median']
    if decoder_available:
        result['first_decoder_loss'] = float(decoder[0])
        result['final_decoder_loss'] = finals[2]
    (args.output_dir/'loss_curve_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
