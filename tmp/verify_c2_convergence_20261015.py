"""Independent C2 continuation audit; no model forward or training is executed.

The selection/status guard precedes all NEW test-bank reads. A capped run can be
audited as resumable/budget-limited, but is never labeled complete/converged.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from verify_direct_context_fm_20261014 import (
    COUNTERS, CP_SHA, archive_sha, close, clusters, interval, read, sha,
    verify_banks, verify_summary,
)
from verify_partial_context_fm_20261015 import allowed, compare_selected, seed_for, verify_cache

REPO = Path(__file__).resolve().parents[1]


def controller_step(old, value, epoch):
    """Independent implementation of the preregistered two-reduction rule."""
    assert np.isfinite(value)
    result = copy.deepcopy(old)
    improved = value < old['best_ES'] - 1e-5
    if improved:
        result['best_ES'], result['best_epoch'] = value, epoch
    meaningful = value <= old['anchor_ES'] * .998
    result['patience'] = 0 if meaningful else old['patience'] + 1
    if meaningful:
        result['anchor_ES'], result['anchor_epoch'] = result['best_ES'], epoch
    event = 'meaningful_improvement' if meaningful else 'waiting'
    if result['patience'] >= 5:
        if old['reductions'] < 2:
            result['reductions'] = old['reductions'] + 1
            result['lrs'] = [rate * .3 for rate in old['lrs']]
            result['patience'] = 0
            result['anchor_ES'], result['anchor_epoch'] = result['best_ES'], epoch
            event = 'reduce_lr'
        else:
            result['converged'] = True
            event = 'validation_plateau_at_min_lr'
    return result, improved, event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=REPO / 'outputs/c2_convergence_20261015/v1')
    args = parser.parse_args()
    study = args.study.resolve()
    # Never open NEW test results above this status/selection guard.
    assert (study / 'validation_selection.json').exists(), 'Selection not yet frozen; leave test closed.'
    selection, summary = read(study / 'validation_selection.json'), read(study / 'summary.json')
    complete = summary['status'] == 'complete'
    if complete:
        assert selection['status'] == 'frozen_before_new_blindtest'
        assert not selection['budget_limited']
    else:
        assert summary['status'] == 'budget_limited'
        assert selection['status'] == 'budget_limited_test_closed' and selection['budget_limited']
        assert summary['new_test_opened'] is False
        assert not (study / 'test/states.json').exists() and not (study / 'test/bank_manifest.json').exists()
    protocol, split, launch = [read(study / p) for p in ('protocol.json', 'split.json', 'launch.json')]
    argv = launch['argv']
    opts = {argv[i]: argv[i+1] for i in range(len(argv)-1) if argv[i].startswith('--')}
    parent = Path(opts['--parent'])
    parent_argv = read(parent / 'launch.json')['argv']
    p_opts = {parent_argv[i]: parent_argv[i+1] for i in range(len(parent_argv)-1) if parent_argv[i].startswith('--')}
    experiment = Path(p_opts['--experiment'])
    refs = dict(checkpoint=experiment / 'snapshot/checkpoints/latest.pt',
                resume=Path(opts.get('--resume', str(parent / 'checkpoints/last_C2.pt'))),
                split=Path(opts['--split']), data_manifest=Path(split['cache_manifest_path']),
                parent_protocol=parent / 'protocol.json', parent_cache=parent / 'frozen_input_manifest.json',
                normalizers=parent / 'normalizers.json', parent_selection=parent / 'validation_selection.json',
                basis=Path(p_opts['--basis']), weak_head=Path(p_opts['--weak-head']),
                scales=Path(p_opts['--scale-reference']), reference_B=parent / 'checkpoints/best_B.pt',
                reference_C2=parent / 'checkpoints/last_C2.pt')
    assert set(refs) == set(protocol['reference_hashes'])
    for name, path in refs.items():
        assert sha(path) == protocol['reference_hashes'][name], name
    assert protocol['reference_hashes']['checkpoint'] == CP_SHA and protocol['trainable_parameters'] == 1_414_528
    assert protocol['seed'] == 20261021
    assert protocol['controller'] == dict(patience=5, meaningful_relative_improvement=.002,
        best_epsilon=1e-5, lr_factor=.3, lr_reductions=2, maximum_global_epoch=int(opts.get('--maximum-epoch', 80)))
    assert sha(study / 'source/tools/continue_partial_context_fm.py') == launch['runner_sha256']
    for name, digest in read(study / 'source_manifest.json').items():
        assert sha(study / 'source' / name) == digest
    if complete:
        for path in read(study / 'import_audit.json').values():
            assert Path(path).resolve().is_relative_to(study / 'source')
    parent_split = read(parent / 'split.json')
    assert split['train'] == parent_split['train'] and split['validation'] == parent_split['validation']
    strip = lambda items: [{k: v for k, v in item.items() if k != 'episode_index'} for item in items]
    assert strip(split['validation_conditions']) == strip(parent_split['validation_conditions'])
    groups = [{e['demand_group'] for e in split[k]} for k in ('train', 'validation', 'test')]
    assert tuple(map(len, groups)) == (64, 16, 16)
    assert not any(groups[i] & groups[j] for i in range(3) for j in range(i))
    assert not groups[2] & {e['demand_group'] for e in parent_split['test']}
    direct = read(REPO / 'outputs/direct_context_fm_20261014/v2_cpu4/split.json')
    assert not groups[2] & {e['demand_group'] for e in direct['test']}
    assert len(split['validation_conditions']) == 8 and len(split['test_conditions']) == 32
    for stage in ('train', 'validation', 'test'):
        for entry in split[stage]:
            assert archive_sha(entry['cache_path']) == entry['cache_content_hash']
    import torch  # Serialization and explicit CPU RNG only, never a model forward.
    original = torch.load(refs['checkpoint'], map_location='cpu', weights_only=True)['training_state']['models']['world_model']
    resumed = torch.load(refs['resume'], map_location='cpu', weights_only=True)
    first_epoch = int(resumed['epoch'])
    assert first_epoch == protocol['resume_epoch'] and protocol['resume_updates'] == first_epoch * 1536
    assert [len(g['params']) for g in resumed['optimizer']['param_groups']] == [18, 4]
    assert len(resumed['optimizer']['state']) == 22
    assert all(float(v['step']) == first_epoch * 1536 and v['step'].device.type == 'cpu'
               for v in resumed['optimizer']['state'].values())
    assert set(resumed['state']['model']) == {k for k in original if allowed(k)}
    parent_protocol = read(refs['parent_protocol'])
    cache_proof = verify_cache(parent, parent_split, parent_protocol, original, .1, torch)
    gates = read(study / 'resume_gates.json')
    assert gates['epoch'] == first_epoch and gates['optimizer_steps'] == first_epoch * 1536 and gates['optimizer_tensors'] == 22
    assert all(gates[k] for k in ('checkpoint_weights_bitexact', 'optimizer_state_bitexact', 'step_tensors_cpu'))
    assert len(gates['forward']) == 2
    new_banks = read(study / 'validation/resume_gate/bank_manifest.json')
    new_rows = read(study / 'validation/resume_gate/states.json')
    verify_banks(split['validation_conditions'][:2], new_rows, new_banks, ('C2',), 4, 100)
    old_banks = resumed.get('validation_banks') or read(parent / f'validation/epoch{first_epoch:03d}/bank_manifest.json')
    old_index = {b['condition_index']: b for b in old_banks}
    for bank, logged in zip(new_banks, gates['forward']):
        old = old_index[bank['condition_index']]
        assert sha(old['path']) == old['sha256']
        with np.load(bank['path']) as actual, np.load(old['path']) as expected:
            np.testing.assert_array_equal(actual['target'], expected['target'])
            np.testing.assert_allclose(actual['C2'], expected['C2'], rtol=1e-5, atol=1e-5)
            close(logged['max_abs'], float(np.max(np.abs(actual['C2']-expected['C2']))))
            assert logged['bit_exact'] == bool(np.array_equal(actual['C2'], expected['C2']))
    parent_best = read(refs['parent_selection'])['best']['C2']
    parent_lrs = [parent_protocol['training']['original_module_lr'], parent_protocol['training']['new_residual_lr']]
    controller = dict(best_ES=parent_best['ES'], best_epoch=parent_best['epoch'],
        anchor_ES=parent_best['reset_ES'], anchor_epoch=parent_best['reset_epoch'], patience=parent_best['patience'],
        reductions=0, lrs=parent_lrs, converged=False)
    initial = read(study / 'initial_controller.json')
    assert initial == resumed.get('controller', controller)
    history = read(study / 'training_history.json')
    assert [r['epoch'] for r in history] == list(range(5, selection['last_epoch']+1))
    expected_resume_controller = copy.deepcopy(controller)
    order_seeds = {}
    for row in history:
        epoch = row['epoch']
        assert not controller['converged'] and row['updates'] == 1536 and row['total_updates'] == epoch * 1536
        assert row['controller_before'] == controller
        close(row['lrs_used'], controller['lrs'])
        report = read(Path(row['validation_summary_path']))
        verify_banks(split['validation_conditions'], report['rows'], report['banks'], ('C2',), 4, 100)
        verify_summary(report['rows'], report, protocol['seed'])
        score = float(np.mean([r['energy_score'] for r in report['rows']]))
        close(row['validation_ES'], score)
        controller, improved, event = controller_step(controller, score, epoch)
        assert row['controller'] == controller and row['best_update'] == improved and row['event'] == event
        order_seeds[epoch] = seed_for(protocol['seed'], 'partial_training_order', epoch)
        if epoch == first_epoch:
            expected_resume_controller = copy.deepcopy(controller)
    if first_epoch > 4:
        assert expected_resume_controller == initial
    assert controller == selection['controller']
    assert selection['total_updates'] == selection['last_epoch'] * 1536
    assert selection['continued_updates'] == (selection['last_epoch']-first_epoch) * 1536
    assert selection['budget_limited'] == (not controller['converged'])
    if controller['converged']:
        assert controller['reductions'] == 2 and controller['patience'] >= 5
        assert selection['reason'] == 'validation_plateau_at_min_lr' and complete
    else:
        assert selection['last_epoch'] == protocol['controller']['maximum_global_epoch']
        assert selection['reason'] == 'maximum_epoch_cap' and not complete
    close(controller['lrs'], np.array(parent_lrs) * .3**controller['reductions'])
    last = torch.load(study / 'checkpoints/last_C2.pt', map_location='cpu', weights_only=True)
    assert last['epoch'] == selection['last_epoch'] and last['controller'] == controller and last['history'] == history
    assert len(last['optimizer']['state']) == 22
    assert [len(g['params']) for g in last['optimizer']['param_groups']] == [18, 4]
    assert all(float(v['step']) == selection['total_updates'] and v['step'].device.type == 'cpu'
               for v in last['optimizer']['state'].values())
    close([g['lr'] for g in last['optimizer']['param_groups']], controller['lrs'])
    best = torch.load(study / 'checkpoints/best_C2.pt', map_location='cpu', weights_only=True)
    compare_selected(best, original, dict(epoch=controller['best_epoch'], ES=controller['best_ES']), 'C2_extended')
    assert set(last['state']) == {'model', 'residual'}
    assert set(last['state']['model']) == {k for k in original if allowed(k)}
    assert set(last['state']['residual']) == set(best['state']['residual'])
    assert sum(v.numel() for v in last['state']['model'].values()) + sum(v.numel() for v in last['state']['residual'].values()) == 1_414_528
    for kind in ('model', 'residual'):
        assert set(best['state'][kind]) == set(last['best_state'][kind])
        for key, value in best['state'][kind].items():
            assert bool(value.equal(last['best_state'][kind][key]))
    gradient = read(study / 'gradient_proof.json')
    assert gradient['frozen_parameter_gradients_absent']
    assert set(gradient['nonzero_gradient_norms']) == {
        'dynamics.flow_head.weight', 'dynamics.reference_proj.weight',
        'dynamics.transformer.layers.5.norm_space.scale', 'residual.output.weight'}
    assert all(np.isfinite(v) and v > 0 for v in gradient['nonzero_gradient_norms'].values())
    result = dict(status='pass' if complete else 'budget_limited_audit_pass_not_complete',
        validation_plateau=controller['converged'], safety_cap_not_convergence=not complete,
        resume_epoch=first_epoch, last_epoch=selection['last_epoch'], total_updates=selection['total_updates'],
        optimizer_resume_full_state_and_cpu_scalars=True, validation_only_controller_recomputed=True,
        lr_reductions=controller['reductions'], lrs=controller['lrs'], selected_best_epoch=controller['best_epoch'],
        selected_best_ES=controller['best_ES'], selected_state_scope_verified=True,
        source_checkpoint_parent_cache_norms_verified=True, fresh_test_groups_disjoint=True,
        optimizer_order_seed_by_global_epoch=order_seeds, cache=cache_proof,
        training_budgets_equal=False, reference_scope='fixed parent B and epoch4 C2; continued C2 consumes more optimization',
        verifier_sha256=sha(__file__))
    if complete:
        assert summary['training_budgets_equal'] is False and summary['continued_only_C2'] is True
        assert summary['frozen_state_unchanged'] and summary['parent_cache_unchanged']
        assert summary['selection'] == selection
        rows = read(study / 'test/states.json')
        variants = ('B_parent', 'C2_epoch4', 'C2_converged')
        verify_banks(split['test_conditions'], rows, read(study / 'test/bank_manifest.json'), variants, 8, 100)
        verify_summary(rows, summary['test'], protocol['seed'])
        for label, a_name, b_name in (('extended_vs_B', 'C2_converged', 'B_parent'),
                                    ('extended_vs_epoch4', 'C2_converged', 'C2_epoch4'),
                                    ('epoch4_vs_B', 'C2_epoch4', 'B_parent')):
            a = {r['condition_index']: r for r in rows if r['arm'] == a_name}
            b = {r['condition_index']: r for r in rows if r['arm'] == b_name}
            assert len(a) == 32 and a.keys() == b.keys()
            for metric, report in summary['pairs'][label].items():
                delta = []
                for index, row in a.items():
                    other = b[index]
                    assert (row['demand_group'], row['episode']) == (other['demand_group'], other['episode'])
                    assert (row[metric] is None) == (other[metric] is None)
                    delta.append(dict(demand_group=row['demand_group'], episode=row['episode'],
                                      difference=None if row[metric] is None else row[metric]-other[metric]))
                interval(clusters(delta, 'difference'), report, protocol['seed'])
        result.update(test_rows=len(rows), test_states=32, test_groups=16, new_test_metrics_verified=True)
    result.update(COUNTERS)
    (study / 'independent_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
