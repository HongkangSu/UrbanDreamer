"""Recompute scores and clustered intervals independently from saved draws."""
import argparse,json
from pathlib import Path
import numpy as np
from verify_direct_context_fm_20261014 import read,sha,close,clusters,interval,verify_banks,verify_summary,COUNTERS


def run(study):
    # Never read a test result before verifying frozen validation selection.
    assert read(study/'independent_training_audit.json')['status']=='pass'
    for arm in ('A','G','B'):
        selection=read(study/arm/'validation_selection.json')
        assert selection['status']=='ready_for_shared_test' and selection['controller']['converged']
    out=study/'shared_test';summary=read(out/'summary.json');protocol=read(out/'test_protocol.json')
    assert summary['status']=='complete' and not summary['training_budgets_equal']
    assert all(summary[k] for k in ['source_unchanged','design_unchanged','parent_cache_unchanged','data_unchanged','selected_checkpoints_unchanged'])
    assert protocol['samples']==8 and protocol['steps']==100 and protocol['horizons']==[1]
    assert protocol['conditions']==32 and protocol['demand_groups']==16
    assert protocol['selections_frozen_before_test']
    hashes=read(out/'selection_hashes.json')
    for name,entry in hashes.items():assert sha(entry['path'])==entry['sha256']==protocol['reference_hashes'][name],name
    for name,digest in read(study/'source_manifest.json').items():assert sha(study/'source'/name)==digest,name
    for path in read(out/'import_audit.json').values():assert Path(path).is_relative_to(study/'source')
    conditions=read(study/'split.json')['test_conditions'];rows=read(out/'states.json');banks=read(out/'bank_manifest.json')
    verify_banks(conditions,rows,banks,('G','A','B','C2'),8,100)
    verify_summary(rows,summary,20261021)
    for name,a,b in [('B_vs_A','B','A'),('A_vs_G','A','G'),('C2_vs_B','C2','B')]:
        aa={r['condition_index']:r for r in rows if r['arm']==a};bb={r['condition_index']:r for r in rows if r['arm']==b}
        differences=[]
        for index,left in aa.items():
            right=bb[index];row={k:left[k] for k in ('demand_group','episode')}
            for metric in summary['pairs'][name]:
                assert (left[metric] is None)==(right[metric] is None)
                row[metric]=None if left[metric] is None else left[metric]-right[metric]
            differences.append(row)
        for metric,stats in summary['pairs'][name].items():interval(clusters(differences,metric),stats,20261021)
    result=dict(status='pass',conditions=32,demand_groups=16,arms=4,checkpoint_selection_before_test=True,
                test_scores_and_cluster_intervals_independently_recomputed=True,verifier_sha256=sha(__file__),**COUNTERS)
    (study/'independent_test_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Source / background-condition controls','',
           'One completed pretrained backbone and one adaptation seed. Partial adaptation with independent validation stopping; consumed update budgets differ. All scores below are lower-is-better.','',
           '| Model | Energy Score | Count CRPS (occupied) | Speed CRPS (occupied) | Selected epoch | Consumed updates |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm in summary['arms']:
        label=arm['arm'];m=arm['metrics'];budget=summary['consumed_budgets'][label]
        lines.append(f"| {label} | {m['energy_score']['estimate']:.6f} | {m['count_crps_occupied']['estimate']:.6f} | {m['speed_crps_occupied']['estimate']:.6f} | {budget['selected_epoch']} | {budget['total_updates']} |")
    lines.extend(['','Paired differences use whole-demand-group bootstrap, 2,000 draws. Negative differences favor the first model. Intervals are unadjusted descriptive 95% intervals.','',
                  '| Comparison | Metric | Difference | 95% interval |','| --- | --- | ---: | --- |'])
    for name in ('B_vs_A','A_vs_G','C2_vs_B'):
        for metric in ('energy_score','count_crps_occupied','speed_crps_occupied'):
            s=summary['pairs'][name][metric]
            lines.append(f"| {name} | {metric} | {s['estimate']:.6f} | [{s['ci_low']:.6f}, {s['ci_high']:.6f}] |")
    lines.extend(['','B/A tests the additional coarse-background condition. A/G changes both source mean and covariance. C2/B changes source and reference beyond the shared background condition.','',
                  'Gaussian G adapts only part of a backbone pretrained with a different source distribution; this is not a ceiling on fully optimized Gaussian FM. No downstream policy benefit is inferred.','',
                  'Independent score, interval, selection and file-integrity audits passed.'])
    (study/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);run(p.parse_args().study.resolve())
