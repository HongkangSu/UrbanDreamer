"""Compare substantive MBPO/Dreamer defects with isolated frozen/current imports.

No learning-protocol or new-argument availability is used as regression evidence.
The fixtures are shared; every production package import must resolve below the
selected source root. All comparisons are bounded deterministic probes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / 'outputs/city_baselines_current_to_dest_1000_20261013_180607/source'
DESTINATION = REPO / 'outputs/baseline_validation_20261015/model_based_regressions'


def fingerprint(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run_child(source, variant):
    source = source.resolve()
    sys.path = [str(source), str(REPO / 'tests')] + [entry for entry in sys.path
        if entry and Path(entry).resolve() not in (REPO, FROZEN, REPO / 'tests')]
    import numpy as np
    import pytest
    import torch
    import urban_dreamer.learning.baselines.common as common
    import urban_dreamer.learning.baselines.dreamer as dreamer
    import urban_dreamer.learning.baselines.mbpo as mbpo
    import test_model_based_stability as regression
    import test_route_model_based_baselines as helper
    from urban_dreamer.learning.replay import MacroTransition

    torch.set_num_threads(1)
    directory = DESTINATION / variant
    directory.mkdir(parents=True, exist_ok=True)
    for module in (common, dreamer, mbpo):
        assert Path(module.__file__).resolve().is_relative_to(source / 'urban_dreamer')
    imports = dict(variant=variant, source=str(source), cwd=str(Path.cwd()),
        executable=sys.executable, sys_path=sys.path, torch_version=torch.__version__,
        production={module.__name__: fingerprint(module.__file__) for module in (common, dreamer, mbpo)},
        fixture=fingerprint(helper.__file__), original_regressions=fingerprint(regression.__file__),
        validator=fingerprint(__file__))
    (directory / 'imports.json').write_text(json.dumps(imports, indent=2) + '\n')

    def seed_replay(method, window, timestamp=0.):
        method.seed_buffer.append_episode([MacroTransition(
            flow=window['flow'], next_flow=window['next_flow'],
            action=window['action'], departure=window['departure'], context=window['context'],
            vehicles=window['vehicles'], active_vehicle_fraction=1., global_reward=-.5,
            time=timestamp, terminal=False, episode=0, step_index=0)],
            total_travel_minutes=.5, total_vehicle_count=1)

    def posterior_seed(patch):
        method = helper.make_method('dreamer')
        flow = method.runner.network.extract_flow({})
        torch.manual_seed(15)
        h, z = method._initial_latent(flow)
        with torch.no_grad():
            zero_h = torch.zeros_like(h)
            encoded = method.world_model.encoder(torch.as_tensor(flow).float().reshape(1, -1))
            mean, std = method.world_model.posterior_params(zero_h, encoded)
        torch.manual_seed(15)
        expected_z = mean + std * torch.randn_like(mean)
        values = dict(h_max_abs=float(h.abs().max()),
            observed_posterior_z_max_abs_error=float((z - expected_z).abs().max()))
        return values, values['h_max_abs'] < 1e-7 and values['observed_posterior_z_max_abs_error'] < 1e-6

    def terminal_frame(patch):
        method = helper.make_method('dreamer')
        batch = regression.one_step_batch(method)
        torch.manual_seed(7)
        _, before = method._world_model_loss(batch)
        changed = copy.deepcopy(batch)
        changed.flow[:, -1].fill_(4.)
        torch.manual_seed(7)
        _, after = method._world_model_loss(changed)
        values = dict(reconstruction_before=before['reconstruction_loss'],
            reconstruction_after=after['reconstruction_loss'],
            reconstruction_delta=after['reconstruction_loss'] - before['reconstruction_loss'])
        return values, values['reconstruction_delta'] > 1.

    def action_reward(patch):
        method = helper.make_method('dreamer')
        batch = regression.one_step_batch(method)
        torch.manual_seed(7)
        _, before = method._world_model_loss(batch)
        changed = copy.deepcopy(batch)
        changed.action.zero_()
        torch.manual_seed(7)
        _, after = method._world_model_loss(changed)
        values = dict(reward_loss_action_one=before['reward_loss'], reward_loss_action_zero=after['reward_loss'],
            abs_reward_loss_delta=abs(after['reward_loss'] - before['reward_loss']))
        return values, values['abs_reward_loss_delta'] > 1e-7

    def joint_kl(patch):
        method = helper.make_method('dreamer')
        batch = regression.one_step_batch(method)
        def prior(h):
            mean = h.new_zeros((len(h), 32))
            return mean, torch.ones_like(mean)
        def posterior(h, encoded):
            mean = h.new_full((len(h), 32), .4)
            return mean, torch.ones_like(mean)
        patch.setattr(method.world_model, 'prior_params', prior)
        patch.setattr(method.world_model, 'posterior_params', posterior)
        _, metrics = method._world_model_loss(batch)
        values = dict(kl=metrics['kl'], expected_joint_kl=32 * .4 ** 2 / 2,
            per_dimension_kl=.4 ** 2 / 2, free_nats=1.)
        return values, abs(values['kl'] - values['expected_joint_kl']) < 1e-5

    def imagined_reward(patch):
        method = helper.make_method('dreamer')
        _, windows = helper.collect_toy_episode(method)
        seed_replay(method, windows[0])
        # Set the existing shared constant for both versions. Do not depend on
        # the new configurable argument to make the frozen probe pass setup.
        patch.setattr(dreamer, 'IMAGINED_HORIZON', 1)
        patch.setattr(method, '_prior_dynamics_step', lambda h, z, *args: (torch.full_like(h, 2.), z))
        patch.setattr(method.world_model, 'predict_reward', lambda h, z, context=None: h[:, :1])
        _, trajectory = method._imagine_rollout(0)
        values = dict(predicted_rewards=trajectory[2], forced_successor_h=2.)
        return values, values['predicted_rewards'] == [2.]

    def real_actor(patch):
        method = helper.make_method('dreamer')
        rows, windows = helper.collect_toy_episode(method)
        before = {name: p.detach().clone() for name, p in method.actor.named_parameters()}
        patch.setattr(method, '_train_world_model', lambda: {})
        patch.setattr(method, '_train_imagination', lambda: dict(
            latent_critic_loss=0., actor_loss=0., imagined_events=0.,
            imagination_seconds=0., actor_update_seconds=0., critic_update_seconds=0.))
        metrics = method.on_episode(rows, windows, episode=0)
        delta = max(float((p.detach() - before[name]).abs().max())
            for name, p in method.actor.named_parameters())
        values = dict(real_decision_count=len(rows), actor_max_abs_parameter_delta=delta,
            reported_actor_loss=metrics['actor_loss'], imagination_disabled=True)
        return values, values['real_decision_count'] > 0 and delta > 0.

    def fresh_synthetic(patch):
        method = helper.make_method('mbpo')
        rows, windows = helper.collect_toy_episode(method)
        # A complete valid stale row avoids interpreting malformed fixture
        # KeyErrors as evidence about actor replay correctness.
        stale = dict(copy.deepcopy(rows[0]), advantage=-.1, sp_prior_weight=0., source='previous_episode')
        method.model_rows.append(stale)
        patch.setattr(method, '_train_model', lambda: {})
        patch.setattr(method, '_policy_updates', lambda *args: dict(actor_loss=0.))
        observed = []
        patch.setattr(method, '_generate_model_rows',
            lambda windows: observed.append(len(method.model_rows)) or 0)
        method.on_episode([], windows, episode=1)
        values = dict(stale_rows_inserted=1, stale_rows_before_new_generation=observed)
        return values, observed == [0]

    def one_actor_update(patch):
        method = helper.make_method('mbpo')
        rows, windows = helper.collect_toy_episode(method)
        calls = []
        patch.setattr(mbpo, 'policy_gradient_step',
            lambda **kwargs: calls.append(len(kwargs['rows'])) or dict(actor_loss=0.))
        method._policy_updates(rows, torch.ones(len(windows)))
        values = dict(policy_gradient_calls=len(calls), rows_per_call=calls)
        return values, len(calls) == 1

    def cohort_terminal(patch, name, timestamp):
        module = dreamer if name == 'dreamer' else mbpo
        method = helper.make_method(name)
        _, windows = helper.collect_toy_episode(method)
        window = dict(windows[0], time=timestamp)
        def arrive(states, **kwargs):
            for state in states:
                state['arrived'] = True
            return SimpleNamespace(action=np.zeros(method.edge_count), departure=np.zeros(method.edge_count),
                active_vehicle_seconds=1.)
        patch.setattr(module, 'advance_imagined_window', arrive)
        if name == 'dreamer':
            seed_replay(method, window, timestamp)
            _, trajectory = method._imagine_rollout(0)
            terminals = trajectory[3]
        else:
            scored = []
            patch.setattr(module, 'global_critic_update',
                lambda **kwargs: (scored.extend(kwargs['windows']) or torch.zeros(len(kwargs['windows'])), {}))
            method._rollout_from_seed(window, 0)
            terminals = [window['terminal'] for window in scored]
        expected = timestamp + 30. >= method.deadline
        values = dict(seed_time=timestamp, deadline=method.deadline,
            cohort_empty=True, scored_terminals=terminals, expected_terminal=expected)
        return values, terminals == [expected]

    probes = [('dreamer_posterior_seed', posterior_seed),
        ('dreamer_terminal_frame_supervision', terminal_frame),
        ('dreamer_action_conditioned_reward_loss', action_reward),
        ('dreamer_joint_free_nats_kl', joint_kl),
        ('dreamer_imagined_successor_reward', imagined_reward),
        ('dreamer_real_actor_anchor', real_actor),
        ('mbpo_fresh_synthetic_actions', fresh_synthetic),
        ('mbpo_single_uncorrected_actor_update', one_actor_update)]
    for name in ('dreamer', 'mbpo'):
        for timestamp in (0., 30.):
            suffix = 'cohort_truncation' if timestamp == 0. else 'deadline_terminal_control'
            probes.append((name + '_' + suffix,
                lambda patch, name=name, timestamp=timestamp: cohort_terminal(patch, name, timestamp)))
    results = []
    for name, probe in probes:
        try:
            with pytest.MonkeyPatch.context() as patch:
                values, passed = probe(patch)
            result = dict(name=name, status='passed' if passed else 'failed', observations=values)
        except Exception:
            result = dict(name=name, status='error', traceback=traceback.format_exc())
        results.append(result)
        print(json.dumps(result), flush=True)
    package_files = {name: str(Path(module.__file__).resolve())
        for name, module in sorted(sys.modules.items())
        if name.startswith('urban_dreamer') and getattr(module, '__file__', None)}
    assert all(Path(path).is_relative_to(source / 'urban_dreamer') for path in package_files.values())
    (directory / 'package_imports.json').write_text(json.dumps(package_files, indent=2) + '\n')
    report = dict(variant=variant, cases=results,
        counts={state: sum(case['status'] == state for case in results)
                for state in ('passed', 'failed', 'error')})
    (directory / 'observations.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if all(case['status'] == 'passed' for case in results) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child-source', type=Path)
    parser.add_argument('--variant')
    args = parser.parse_args()
    if args.child_source:
        return run_child(args.child_source, args.variant)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    variants = {}
    for variant, source in (('frozen', FROZEN), ('current', REPO)):
        directory = DESTINATION / variant
        directory.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = [sys.executable, str(Path(__file__).resolve()), '--child-source', str(source), '--variant', variant]
        result = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, timeout=180)
        (directory / 'validation.log').write_text(result.stdout + result.stderr)
        data_path = directory / 'observations.json'
        variants[variant] = dict(source=str(source), returncode=result.returncode, command=command,
            imported=json.loads((directory / 'imports.json').read_text()) if (directory / 'imports.json').exists() else None,
            results=json.loads(data_path.read_text()) if data_path.exists() else None)
    verified = (variants['frozen']['returncode'] == 1
        and variants['frozen']['results']['counts'] == dict(passed=2, failed=10, error=0)
        and variants['current']['returncode'] == 0
        and variants['current']['results']['counts'] == dict(passed=12, failed=0, error=0))
    comparisons = []
    if all(data['results'] for data in variants.values()):
        for before, after in zip(variants['frozen']['results']['cases'], variants['current']['results']['cases']):
            assert before['name'] == after['name']
            comparisons.append(dict(name=before['name'], frozen=before['status'], current=after['status'],
                frozen_observations=before.get('observations'), current_observations=after.get('observations')))
    report = dict(verified=verified, variants=variants, comparisons=comparisons,
        limitation='Deterministic bounded implementation-defect probes, including controlled latent/cohort injections. '
                   'No convergence, aggregate traffic reward, or runtime speed claim.')
    (DESTINATION / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(verified=verified, summary=str(DESTINATION / 'summary.json'),
        results={name: data['results']['counts'] if data['results'] else None for name, data in variants.items()}), indent=2))
    return 0 if verified else 1


if __name__ == '__main__':
    raise SystemExit(main())
