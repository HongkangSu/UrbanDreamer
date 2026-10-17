from pathlib import Path
import json,sys,hashlib
root=Path.cwd();study=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else root/'outputs/c2_convergence_20261015/v1'
read=lambda p:json.loads(p.read_text())
a=read(study/'summary.json');v=read(study/'independent_verification.json');assert a['status']=='complete' and v['status']=='pass' and v['validation_plateau']
p=read(study/'protocol.json');selection=a['selection'];history=read(study/'training_history.json')
metrics=['energy_score','energy_score_u','count_crps_occupied','speed_crps_occupied','count_mae_occupied','speed_mae_occupied','count_coverage90_occupied','speed_coverage90_occupied','count_width90_occupied','speed_width90_occupied','total_count_absolute_error','total_count_bias']
result=dict(status='complete_validation_plateau',experiment='C2 continuation from epoch4 to validation plateau with complete AdamW restoration',checkpoint_sha256=p['reference_hashes']['checkpoint'],completed_pretrain_updates=49636,protocol=p,
 comparison_semantics=dict(B_parent='sourceZ+epsilon,referenceZ,backgroundb suppliedtoDirectResidual head',C2='sourceZ+b+epsilon,referenceZ+b,samebackgroundhint',noise='sigma*U16*Gaussian;currentstatecenteredgraphGaussian,NOTstandardzero-meanisotropicGaussian',zero_mean_legacy_name='zero prior increment,NOTzero fullsource mean',what_is_tested='Incremental source/reference use after bothmodelsreceivebackground; notpriorpresencevsabsence'),interpretation=dict(continued_training='EnergyScore improves versusepoch4C2, butdidnotestablishoverallwinagainstfixedB.',joint_score='ExtendedminusB ES+.29235 CI[-.01452,.62603],notclearlydifferent.',count_and_mass='OccupiedcountCRPS remainsworse andtotalcountabsoluteerror increases; nooverallroutinggainclaim.',speed='SpeedsCRPS pointimproves butextendedvsB CIcrosseszero.',convergence='Validationplateauatglobalepoch25aftertwoLRcuts;selectedbestepoch10;partialadaptationonly'),selection=selection,optimization=dict(resume_epoch=p['resume_epoch'],last_global_epoch=selection['last_epoch'],last_total_adaptation_updates=selection['total_updates'],best_global_epoch=selection['controller']['best_epoch'],best_updates=selection['controller']['best_epoch']*1536,learning_rate_reductions=[dict(epoch=r['epoch'],next_lrs=r['controller']['lrs']) for r in history if r['event']=='reduce_lr']),
 results={r['arm']:{m:r['metrics'][m] for m in metrics if m in r['metrics']} for r in a['test']['arms']},
 contrasts={k:{m:d[m] for m in metrics if m in d} for k,d in a['pairs'].items()},
 scope=['Validation plateau of the same1,414,528-parameter partial adaptation family, not global optimality or full backbone joint convergence.',
 'Only C2 trained longer; B and epoch4C2 remain fixed references. Training budgets are unequal.',
 'Same training64groups/768windows/fourtime-noise views. No additional SUMO transitions.',
 'New test16groups/32states used only after plateau and checkpoint selection. One backbone/adaptationseed.',
 'No regression, policy or out-of-distribution-demand benefit established by this comparison.'],
 resume_gates=read(study/'resume_gates.json'),independent_verification=v,
 validation=dict(focused_tests='16passed (continuation optimizer/controller plus existing partial and checkpoint scope)',compile='passed',source='frozen snapshot; parent cache and norms unchanged'),
 runtime=dict(elapsed_seconds=a['elapsed_seconds'],max_gpu_allocated_mib=a['max_gpu_allocated_mib']),
 artifacts=dict(root=str(study.relative_to(root)),summary=str((study/'summary.json').relative_to(root)),curve=str((study/'convergence_curve.png').relative_to(root))))
if (study/'fair_crps_diagnostic.json').exists():result['fair_crps_diagnostic']=read(study/'fair_crps_diagnostic.json')
if (study/'h4_gate.json').exists():result['h4_gate']=read(study/'h4_gate.json')
if (study/'h4_followup/summary.json').exists():result['h4_followup']=read(study/'h4_followup/summary.json')
if (study/'parent_coarse_accuracy_analysis.json').exists():
 d=read(study/'parent_coarse_accuracy_analysis.json');result['parent_coarse_accuracy_summary']={k:v for k,v in d.items() if k!='rows'}
for name in ['convergence_curve.png','convergence_curve.pdf']:
 if (study/name).exists():result['artifacts'][name+'_sha256']=hashlib.sha256((study/name).read_bytes()).hexdigest()
(root/'docs/c2_convergence_20261015.json').write_text(json.dumps(result,indent=2)+'\n');print('docs/c2_convergence_20261015.json')
