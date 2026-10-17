"""Independent selection audit. Never reads shared test predictions."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from verify_c2_convergence_20261015 import controller_step
from verify_direct_context_fm_20261014 import read,sha,close,verify_banks,verify_summary,COUNTERS


def run(study):
    source=study/'source';manifest=read(study/'source_manifest.json')
    for name,digest in manifest.items():assert sha(source/name)==digest,name
    design=read(study/'design.json');split=read(study/'split.json')
    assert sha(design['reference_C2']['path'])==design['reference_C2']['sha256']
    selections={}
    for arm in ('A','G','B'):
        d=study/arm;selection=read(d/'validation_selection.json')
        assert selection['status']=='ready_for_shared_test' and not selection['budget_limited']
        assert selection['controller']['converged']
        selections[arm]=selection
    launch=read(study/'launch.json')['arms']['B']['argv'];parent=Path(launch[launch.index('--parent')+1])
    parent_split=read(parent/'split.json')
    for name in ('train','validation'):assert split[name]==parent_split[name]
    strips=lambda x:[{k:v for k,v in i.items() if k!='episode_index'} for i in x]
    assert strips(split['validation_conditions'])==strips(parent_split['validation_conditions'])
    groups=[{e['demand_group'] for e in split[s]} for s in ('train','validation','test')]
    assert tuple(map(len,groups))==(64,16,16)
    assert not any(groups[i]&groups[j] for i in range(3) for j in range(i))
    for prior in [parent_split,read(study.parents[1]/'c2_convergence_20261015/v1/split.json')]:
        assert not groups[2]&{e['demand_group'] for e in prior['test']}
    counts={}
    import torch
    torch.set_num_threads(2)
    for arm,selection in selections.items():
        d=study/arm;protocol=read(d/'protocol.json');norm=read(d/'normalizers.json')
        assert protocol['arm']==arm and protocol['trainable_parameters']==1414528
        assert protocol['background_condition']==(arm=='B') and protocol['reference']=='Z'
        assert norm['fit_split']=='trainonly'
        for p in read(d/'import_audit.json').values():assert Path(p).is_relative_to(source)
        proof=read(d/'gradient_proof.json');assert proof['frozen_parameter_gradients_absent']
        assert all(np.isfinite(v) and v>0 for v in proof['nonzero_gradient_norms'].values())
        controller=read(d/'initial_controller.json');first_epoch=4 if arm=='B' else 0
        initial=read(d/f'validation/epoch{first_epoch:03d}_summary.json')
        verify_banks(split['validation_conditions'],initial['rows'],initial['banks'],(arm,),4,100)
        verify_summary(initial['rows'],initial,20261021)
        if arm=='B':
            old=read(parent/'validation_selection.json')['best']['B']
            assert controller==dict(best_ES=old['ES'],best_epoch=old['epoch'],anchor_ES=old['reset_ES'],anchor_epoch=old['reset_epoch'],patience=3,reductions=0,lrs=[.0001,.001],converged=False)
            gates=read(d/'resume_gates.json');assert gates['optimizer_state_bitexact']
            assert len(gates['forward'])==8 and all(v['max_abs']==0 for v in gates['forward'])
        else:
            close(controller['best_ES'],np.mean([r['energy_score'] for r in initial['rows']]))
            assert controller['best_epoch']==0 and controller['patience']==0
        history=read(d/'training_history.json')
        assert [r['epoch'] for r in history]==list(range(first_epoch+1,selection['last_epoch']+1))
        for row in history:
            assert not controller['converged'];close(row['lrs_used'],controller['lrs'])
            assert row['updates']==1536 and row['total_updates']==row['epoch']*1536
            report=read(d/f"validation/epoch{row['epoch']:03d}_summary.json")
            verify_banks(split['validation_conditions'],report['rows'],report['banks'],(arm,),4,100)
            verify_summary(report['rows'],report,20261021)
            es=float(np.mean([r['energy_score'] for r in report['rows']]))
            close(row['validation_ES'],es)
            controller,improved,event=controller_step(controller,es,row['epoch'])
            assert row['controller']==controller and row['best_update']==improved and row['event']==event
        assert controller==selection['controller'] and controller['reductions']==2 and controller['patience']>=5
        last=torch.load(d/f'checkpoints/last_{arm}.pt',map_location='cpu',weights_only=True)
        best=torch.load(d/f'checkpoints/best_{arm}.pt',map_location='cpu',weights_only=True)
        assert last['epoch']==selection['last_epoch'] and last['controller']==controller and last['history']==history
        assert selection['total_updates']==selection['last_epoch']*1536
        assert selection['continued_updates']==(selection['last_epoch']-first_epoch)*1536
        assert all(int(v['step'])==selection['total_updates'] for v in last['optimizer']['state'].values())
        assert best['epoch']==controller['best_epoch'];close(best['validation_ES'],controller['best_ES'])
        assert sum(v.numel() for g in best['state'].values() for v in g.values())==1414528
        for kind in ('model','residual'):
            for name,v in best['state'][kind].items():assert v.equal(last['best_state'][kind][name])
        counts[arm]=dict(last_epoch=last['epoch'],best_epoch=best['epoch'],best_validation_ES=best['validation_ES'],total_updates=selection['total_updates'])
    for name,digest in read(parent/'frozen_input_manifest.json').items():assert sha(parent/'frozen_inputs'/name)==digest
    assert sha(study/'G/gaussian_noise.npy')==read(study/'G/normalizers.json')['gaussian_source_audit']['sha256']
    result=dict(status='pass',arms=counts,test_prediction_files_read=False,verifier_sha256=sha(__file__),**COUNTERS)
    (study/'independent_training_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);run(p.parse_args().study.resolve())
