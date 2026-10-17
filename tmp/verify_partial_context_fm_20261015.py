"""Independent partial-adaptation arithmetic and integrity audit, at completion.

No model code is imported. NumPy bank/statistic helpers come from the earlier
independent verifier, whose main() is not executed. Torch is used for serialized
checkpoints and reproducing the original CPU coefficient/time RNG only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from verify_direct_context_fm_20261014 import (
    COUNTERS, CP_SHA, archive_sha, close, clusters, interval, read, sha,
    verify_banks, verify_summary,
)

ARMS = ('B', 'C1', 'C2')
PREFIXES = ('dynamics.transformer.layers.5.', 'dynamics.reference_proj.', 'dynamics.flow_head.')
REPO = Path(__file__).resolve().parents[1]


def allowed(name):
    return name.startswith(PREFIXES)


def seed_for(*values):
    return int(hashlib.sha256(':'.join(map(str, values)).encode()).hexdigest()[:8], 16)


def compare_selected(checkpoint, reference, best, name):
    state = checkpoint['state']
    assert set(state) == {'model', 'residual'}
    expected = {k for k in reference if allowed(k)}
    assert set(state['model']) == expected
    assert checkpoint['epoch'] == best['epoch']
    close(checkpoint['validation_ES'], best['ES'])
    assert sum(v.numel() for v in state['model'].values()) == 1_348_672
    for key, value in state['model'].items():
        assert value.shape == reference[key].shape and value.dtype == reference[key].dtype
        assert bool(value.isfinite().all()), (name, key)
        if best['epoch'] == 0:
            assert bool(value.equal(reference[key])), (name, 'epoch0 did not retain pretrained weights', key)
    shapes = {'hidden.weight': (128, 449), 'hidden.bias': (128,),
              'output.weight': (64, 128), 'output.bias': (64,)}
    assert set(state['residual']) == set(shapes)
    assert sum(v.numel() for v in state['residual'].values()) == 65_856
    for key, value in state['residual'].items():
        assert tuple(value.shape) == shapes[key] and bool(value.isfinite().all())
    if best['epoch'] == 0:
        assert not bool(state['residual']['output.weight'].any())
        assert not bool(state['residual']['output.bias'].any())


def verify_cache(study, split, protocol, original_state, noise_scale, torch):
    folder = study / 'frozen_inputs'
    assert (study / 'frozen_input_manifest.json').exists()
    manifest = read(study / 'frozen_input_manifest.json')
    assert set(manifest) == {key + '.npy' for key in ('z', 'target', 'b', 'condition', 'action', 'example_window', 'time', 'noise')}
    for name, digest in manifest.items():
        assert sha(folder / name) == digest
    arrays = {name.removesuffix('.npy'): np.load(folder / name, mmap_mode='r') for name in manifest}
    nw, roads = 768, 853
    for key, shape in dict(z=(nw, 1, roads, 64), target=(nw, 1, roads, 64), b=(nw, 1, roads, 64),
                           condition=(nw, 1, roads, 256), action=(nw, 1, roads)).items():
        assert arrays[key].shape == shape and arrays[key].dtype == np.float32
    assert arrays['time'].shape == (3072, 1) and arrays['noise'].shape == (3072, 1, 16, 64)
    assert arrays['example_window'].shape == (3072,) and arrays['example_window'].dtype == np.int64
    metadata = read(study / 'draw_schedule.json')
    cursor = offset = schedule_index = 0
    for entry in split['train']:
        starts = entry['starts']
        assert len(starts) == 12
        for quarter in range(4):
            actual = metadata[schedule_index]
            assert actual['stage'] == 'train' and actual['cache_path'] == entry['cache_path'] and actual['quarter'] == quarter
            assert len(actual['draws']) == len(starts)
            for local, start in enumerate(starts):
                ts = seed_for(protocol['seed'], 'train', 'uniform_time', entry['cache_path'], int(start), quarter)
                ns = seed_for(protocol['seed'], 'train', 'source_noise', entry['cache_path'], int(start), quarter)
                uniform = float(torch.rand((), generator=torch.Generator().manual_seed(ts)))
                expected_time = np.float32((quarter + uniform) / 4)
                expected_noise = torch.randn(1, 16, 64, generator=torch.Generator().manual_seed(ns)).numpy()
                assert arrays['example_window'][cursor] == offset + local
                np.testing.assert_array_equal(arrays['time'][cursor], np.array([expected_time], dtype=np.float32))
                np.testing.assert_array_equal(arrays['noise'][cursor], expected_noise)
                record = actual['draws'][local]
                assert record == dict(start=start, quarter=quarter, time=float(expected_time), time_seed=ts, noise_seed=ns)
                cursor += 1
            schedule_index += 1
        offset += len(starts)
    assert cursor == 3072 and offset == 768 and schedule_index == len(metadata) == 256
    normalizers = read(study / 'normalizers.json')
    assert normalizers['fit_split'] == 'trainonly' and normalizers['centering'] is False
    for key in ('z', 'b'):
        expected = sum(np.square(np.asarray(arrays[key][i:i+12]), dtype=np.float64).sum()
                       for i in range(0, nw, 12))
        close(normalizers['sums'][key], expected)
        assert normalizers['counts'][key] == nw * roads * 64
    # Reconstruct the original rank16 source noise; no network inference.
    basis = original_state['dynamics.graph_prior.basis'].numpy().astype(np.float32)
    assert basis.shape == (roads, 16)
    energy = 0.
    for i in range(0, len(arrays['time']), 12):
        windows = arrays['example_window'][i:i+12]
        z, target, b = [np.asarray(arrays[key][windows]) for key in ('z', 'target', 'b')]
        t = np.asarray(arrays['time'][i:i+12])[:, :, None, None]
        epsilon = np.float32(noise_scale) * np.matmul(basis, np.asarray(arrays['noise'][i:i+12]))
        for source_on in (False, True):
            mean = z + b if source_on else z
            relative = (1-t) * (mean + epsilon) + t * target - mean
            energy += np.square(relative, dtype=np.float64).sum()
    # CPU/GPU FP32 matmul rounding differs; the algebra/scale/count must agree.
    np.testing.assert_allclose(normalizers['sums']['relative'], energy, rtol=2e-6, atol=1e-5)
    assert normalizers['counts']['relative'] == 3072 * roads * 64 * 2
    for key in ('z', 'b', 'relative'):
        close(normalizers['scales'][key], max(np.sqrt(normalizers['sums'][key] / normalizers['counts'][key]), 1e-8))
    return dict(window_rows=nw, augmented_graphs=cursor, shared_draws_reproduced=True,
                training_only_normalizers_recomputed=True, cache_files=len(manifest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=REPO / 'outputs/partial_context_fm_20261015/v1')
    args = parser.parse_args()
    study = args.study.resolve()
    # Never open partial-study test results before validation selection exists.
    assert (study / 'validation_selection.json').exists(), 'Selection not frozen; leave test closed.'
    selected = read(study / 'validation_selection.json')
    assert selected['status'] == 'frozen_before_new_blindtest'
    summary = read(study / 'summary.json')
    assert summary['status'] == 'complete', 'Run verifier only at completion.'
    protocol, split, launch = [read(study / p) for p in ('protocol.json', 'split.json', 'launch.json')]
    seed = protocol['seed']
    argv = launch['argv']
    opts = {argv[i]: argv[i+1] for i in range(len(argv)-1) if argv[i].startswith('--')}
    pilot_dir, experiment = Path(opts['--pilot']), Path(opts['--experiment'])
    refs = dict(checkpoint=experiment / 'snapshot/checkpoints/latest.pt', split=Path(opts['--split']),
                cache_manifest=Path(split['cache_manifest_path']), pilot=pilot_dir / 'pilot_summary.json',
                basis=Path(opts['--basis']), weak_head=Path(opts['--weak-head']), scale_reference=Path(opts['--scale-reference']))
    assert set(refs) == set(protocol['reference_hashes'])
    for name, path in refs.items():
        assert sha(path) == protocol['reference_hashes'][name], name
    assert protocol['reference_hashes']['checkpoint'] == CP_SHA
    assert protocol['reference_hashes']['cache_manifest'] == launch['initial_data_manifest_sha256']
    assert 'entries' in read(refs['cache_manifest'])  # Original dataset manifest was not overwritten.
    assert refs['cache_manifest'].resolve() != (study / 'frozen_input_manifest.json').resolve()
    audit = read(study / 'data_split_audit.json')
    assert audit['split_sha256'] == protocol['reference_hashes']['split']
    assert audit['subset_manifest_sha256'] == protocol['reference_hashes']['cache_manifest']
    assert sha(study / 'source/tools/run_partial_context_fm.py') == launch['runner_sha256']
    for name, digest in read(study / 'source_manifest.json').items():
        assert sha(study / 'source' / name) == digest
    for path in read(study / 'import_audit.json').values():
        assert Path(path).resolve().is_relative_to(study / 'source')
    assert all(summary[k] for k in ('frozen_state_unchanged', 'source_unchanged', 'data_unchanged'))
    assert protocol['arms'] == {a: dict(source_on=s, reference_on=r, hint='same frozen state-only b')
                                for a, s, r in (('B', False, False), ('C1', True, False), ('C2', True, True))}
    groups = [{e['demand_group'] for e in split[k]} for k in ('train', 'validation', 'test')]
    assert tuple(map(len, groups)) == (64, 16, 16)
    assert not any(groups[i] & groups[j] for i in range(3) for j in range(i))
    direct_split = read(REPO / 'outputs/direct_context_fm_20261014/v2_cpu4/split.json')
    assert not groups[2] & {e['demand_group'] for e in direct_split['test']}
    assert split['train'] == direct_split['train'] and split['validation'] == direct_split['validation']
    strip = lambda items: [{k: v for k, v in item.items() if k != 'episode_index'} for item in items]
    assert strip(split['validation_conditions']) == strip(direct_split['validation_conditions'])
    assert len(split['validation_conditions']) == 8 and len(split['test_conditions']) == 32
    assert len({e['demand_group'] for e in split['validation_conditions']}) == 4
    for stage in ('train', 'validation', 'test'):
        for e in split[stage]:
            assert archive_sha(e['cache_path']) == e['cache_content_hash']
    pilot = read(refs['pilot'])
    assert pilot['status'] == 'pilot_complete' and pilot['optimizer_steps'] == 0
    assert all(pilot[k] for k in ('parameters_unchanged', 'gradients_cleared', 'rng_unchanged'))
    for key in ('checkpoint', 'split', 'basis', 'weak_head'):
        assert pilot['reference_hashes'][key] == protocol['reference_hashes'][key]
    for arm in ARMS:
        records = [r for r in pilot['records'] if r['arm'] == arm]
        assert len(records) == 3
        close(np.median([r['seconds'] for r in records]), pilot['median_seconds_per_arm'][arm])
    gates = read(study / 'initial_gates.json')
    assert gates['same_initial_trainable_weights'] and gates['new_heads_zero_output']
    assert gates['all_arms_parameters'] == protocol['trainable_parameters_per_arm'] == summary['trainable_parameters_per_arm'] == 1_414_528
    gradient = read(study / 'gradient_proof.json')
    assert set(gradient) == set(ARMS)
    for proof in gradient.values():
        assert proof['frozen_parameter_gradients_absent']
        assert set(proof['nonzero_gradient_norms']) == {
            'dynamics.flow_head.weight', 'dynamics.reference_proj.weight',
            'dynamics.transformer.layers.5.norm_space.scale', 'residual.output.weight'}
        assert all(np.isfinite(v) and v > 0 for v in proof['nonzero_gradient_norms'].values())
    import torch  # No model is instantiated and no model forward is executed.
    raw = torch.load(refs['checkpoint'], map_location='cpu', weights_only=True)
    original = raw['training_state']['models']['world_model']
    noise_scale = float(re.search(r'^prior_noise_scale\s*=\s*([0-9.eE+-]+)',
                                 (experiment / 'snapshot/run_config.toml').read_text(), re.M).group(1))
    cache_proof = verify_cache(study, split, protocol, original, noise_scale, torch)
    training = protocol['training']
    assert training['batch_whole_graphs'] == 2
    assert training['minimum_epochs'] == 3 and training['maximum_epochs'] == 4
    assert training['validation_every_epochs'] == 1
    history = read(study / 'training_history.json')
    epochs = len(history)
    assert 3 <= epochs <= 4 and [r['epoch'] for r in history] == list(range(1, epochs+1))
    updates = 3072 // 2
    assert all(r['updates_per_arm'] == updates and set(r['train_CFM_MSE']) == set(ARMS) for r in history)
    assert selected['actual_equal_epochs'] == epochs and selected['updates_per_arm'] == epochs * updates
    initial = read(study / 'validation/epoch000_summary.json')
    verify_banks(split['validation_conditions'], initial['rows'], initial['banks'], ARMS, 4, 100)
    verify_summary(initial['rows'], initial, seed)
    best = {}
    for arm in ARMS:
        es = float(np.mean([r['energy_score'] for r in initial['rows'] if r['arm'] == arm]))
        best[arm] = dict(ES=es, epoch=0, reset_ES=es, reset_epoch=0, patience=0)
    for row in history:
        report = read(study / f'validation/epoch{row["epoch"]:03d}_summary.json')
        verify_banks(split['validation_conditions'], report['rows'], report['banks'], ARMS, 4, 100)
        verify_summary(report['rows'], report, seed)
        assert set(row['integrated_validation']) == set(ARMS)
        for arm in ARMS:
            es = float(np.mean([r['energy_score'] for r in report['rows'] if r['arm'] == arm]))
            old = best[arm].copy()
            improved, meaningful = es < old['ES'] - 1e-5, es <= old['reset_ES'] * .998
            if improved:
                best[arm].update(ES=es, epoch=row['epoch'])
            if meaningful:
                best[arm].update(reset_ES=best[arm]['ES'], reset_epoch=row['epoch'], patience=0)
            else:
                best[arm]['patience'] += 1
            logged = row['integrated_validation'][arm]
            close(logged['ES'], es)
            assert logged['best_update'] == improved and logged['meaningful_reset'] == meaningful
            assert logged['selection'] == best[arm]
        if row['epoch'] != epochs:
            assert row['epoch'] < 3 or not all(b['patience'] >= 3 for b in best.values())
    reason = 'all_meaningful_ES_patience3' if all(b['patience'] >= 3 for b in best.values()) else 'maximum_epoch_cap'
    assert selected['stop_reason'] == reason and selected['budget_limited'] == (reason == 'maximum_epoch_cap')
    if selected['budget_limited']:
        assert epochs == 4
    assert best == selected['best'] == summary['selection']['best']
    assert summary['selection'] == selected
    for arm in ARMS:
        checkpoint = torch.load(study / f'checkpoints/best_{arm}.pt', map_location='cpu', weights_only=True)
        compare_selected(checkpoint, original, best[arm], arm)
        last = torch.load(study / f'checkpoints/last_{arm}.pt', map_location='cpu', weights_only=True)
        assert last['epoch'] == epochs
        assert set(last['state']['model']) == {k for k in original if allowed(k)}
        optimizer = last['optimizer']
        assert [len(g['params']) for g in optimizer['param_groups']] == [18, 4]
        close([g['lr'] for g in optimizer['param_groups']], [training['original_module_lr'], training['new_residual_lr']])
        assert all(g['weight_decay'] == .0001 for g in optimizer['param_groups'])
        assert len(optimizer['state']) == 22
        assert all(float(v['step']) == epochs * updates for v in optimizer['state'].values())

    # Final results are opened only after the validation selection is reconstructed.
    test_rows = read(study / 'test/states.json')
    verify_banks(split['test_conditions'], test_rows, read(study / 'test/bank_manifest.json'), ARMS, 8, 100)
    verify_summary(test_rows, summary['test'], seed)
    for label, left, right in (('source_effect', 'C1', 'B'), ('reference_effect', 'C2', 'C1'), ('joint_effect', 'C2', 'B')):
        a = {r['condition_index']: r for r in test_rows if r['arm'] == left}
        b = {r['condition_index']: r for r in test_rows if r['arm'] == right}
        assert len(a) == 32 and a.keys() == b.keys()
        for metric, report in summary['pairs'][label].items():
            differences = []
            for index, row in a.items():
                other = b[index]
                assert (row['demand_group'], row['episode']) == (other['demand_group'], other['episode'])
                assert (row[metric] is None) == (other[metric] is None)
                differences.append(dict(demand_group=row['demand_group'], episode=row['episode'],
                    delta=None if row[metric] is None else row[metric]-other[metric]))
            interval(clusters(differences, 'delta'), report, seed)
    result = dict(status='pass', independent_numpy_bank_scores=True, independent_group_bootstrap=True,
        validation_only_selection=True, equal_actual_epochs=epochs, equal_actual_updates_per_arm=epochs * updates,
        trainable_parameters_per_arm=1_414_528, selected_state_scope_verified=True,
        reference_gradient_through_frozen_prefix_verified=True, original_data_manifest_not_overwritten=True,
        fresh_test_disjoint_direct_v2=True, checkpoint_sha256=CP_SHA, integration_steps=100,
        budget_limited=selected['budget_limited'], best=best, test_rows=len(test_rows), test_states=32,
        adaptation_effect_scope='equal-budget independently adapted variants; not within-identical-weights causal intervention',
        cache=cache_proof, verifier_sha256=sha(__file__),
        numerical_helper_sha256=sha(Path(__file__).with_name('verify_direct_context_fm_20261014.py')), **COUNTERS)
    (study / 'independent_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
