from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd();study=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else root/'outputs/c2_convergence_20261015/v1';parent=root/'outputs/partial_context_fm_20261015/v2_gpu0'
read=lambda p:json.loads(p.read_text())
initial=read(parent/'validation/epoch000_summary.json');initial_es=next(r['metrics']['energy_score']['estimate'] for r in initial['arms'] if r['arm']=='C2')
old=read(parent/'training_history.json');new=read(study/'training_history.json')
epochs=[0]+[r['epoch'] for r in old]+[r['epoch'] for r in new]
es=[initial_es]+[r['integrated_validation']['C2']['ES'] for r in old]+[r['validation_ES'] for r in new]
loss_epochs=[r['epoch'] for r in old]+[r['epoch'] for r in new]
losses=[r['train_CFM_MSE']['C2'] for r in old]+[r['train_CFM_MSE'] for r in new]
baseline=read(parent/'validation_selection.json')['best']['B']['ES'];x=np.asarray(epochs)*1.536
fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True,gridspec_kw={'height_ratios':[2,1]},constrained_layout=True)
axes[0].plot(x,es,'o-',color='#2563eb',label='C2 validation ES',markersize=3)
axes[0].plot(x,np.minimum.accumulate(es),'--',color='#059669',label='Best validation ES')
axes[0].axhline(baseline,color='#52525b',ls=':',label='Fixed B validation reference')
axes[0].set_ylabel('Energy Score (lower is better)')
if (study/'validation_selection.json').exists():
 c=read(study/'validation_selection.json')['controller'];axes[0].scatter([c['best_epoch']*1.536],[c['best_ES']],marker='*',s=120,color='#f59e0b',edgecolors='black',linewidths=.5,label='Selected checkpoint',zorder=5)
axes[0].legend(fontsize=8)
axes[1].plot(np.asarray(loss_epochs)*1.536,losses,'o-',color='#7c3aed',markersize=3)
axes[1].set_yscale('log');axes[1].set_ylabel('Training CFM MSE');axes[1].set_xlabel('Total adaptation updates (thousands)')
for ax in axes:
 ax.axvline(4*1.536,color='#9ca3af',ls='--',lw=.8);ax.grid(alpha=.2)
 for row in new:
  if row['event']=='reduce_lr':ax.axvline((row['epoch']+.5)*1.536,color='#ef4444',ls=':',lw=.9)
axes[0].set_title('C2 continuation: fixed validation states and sampling noise')
fig.text(.50,-.01,'Dashed gray: resume at epoch 4. Dotted red: learning-rate reductions. One adaptation run.',ha='center',fontsize=8)
for suffix in ('png','pdf'):fig.savefig(study/f'convergence_curve.{suffix}',dpi=180,bbox_inches='tight')
plt.close(fig);print(study/'convergence_curve.png')
