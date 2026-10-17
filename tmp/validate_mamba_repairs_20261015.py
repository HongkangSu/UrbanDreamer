"""Compare bounded MAMBA regressions using isolated frozen/current imports."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / 'outputs/city_baselines_current_to_dest_1000_20261013_180607/source'
DESTINATION = REPO / 'outputs/baseline_validation_20261015/mamba_regressions'
TEST = REPO / 'tests/test_mamba_stability.py'
SELECTION = ('known_physical_deadline or partial_agent_chunk or custom_kl_threshold '
             'or forced_actions_do_not_dilute or forced_only_batch')


def fingerprint(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def observe(regression, helper, mamba, pytest, torch):
    observations = {}
    for deadline in (90., 300.):
        method = helper.make_method()
        rows, windows = helper.episode(method)
        method.runner.deadline = deadline
        data = method._pack_episode(rows, windows, 1)
        with pytest.MonkeyPatch.context() as monkeypatch:
            helper.fixed_flow(method, monkeypatch)
            monkeypatch.setattr(method.global_model.reward_model, 'forward',
                lambda x: x.new_full((*x.shape[:-1], 1), -1.))
            monkeypatch.setattr(method.global_model.continuation_model, 'forward',
                lambda x: x.new_full((*x.shape[:-1], 1), -100.))
            monkeypatch.setattr(method.critic, 'forward',
                lambda x: x.new_full((*x.shape[:-1], 1), -10.))
            state = method.global_model.initial_state(1, 1, device=method.device)
            _, _, returns = method._imagined_rollout(data, state, 0, {'a'})
        expected = ([-3., -2., -1.] if deadline == 90. else
            (-torch.arange(mamba.IMAGINATION_HORIZON, 0, -1).float() - 10.).tolist())
        observations['physical_deadline_' + str(int(deadline))] = dict(
            returns=returns.tolist(), expected_returns=expected,
            passes=bool(torch.allclose(returns, torch.tensor(expected))),
            reward_per_window=-1., continuation_logit=-100., horizon_bootstrap_value=-10.)

    for name, counts, forced, advantages, expected in (
        ('partial_chunks', (32, 1), False, (1., -1.), -31. / 33.),
        ('forced_rows_in_mixed_batch', (1, 32), True, (1., -1.), -1.)):
        method = regression.policy_method()
        losses = []
        method._step = lambda optimizer, loss, parameters: losses.append(float(loss.detach()))
        method._ppo_update([regression.record(method, counts[0], 0),
            regression.record(method, counts[1], 1, forced=forced)], torch.tensor(advantages))
        observations[name] = dict(chunk_sizes=counts, advantages=advantages,
            optimizer_step_losses=losses, expected_loss=expected,
            passes=bool(losses and all(abs(loss - expected) < 1e-6 for loss in losses)))

    for target_kl, expected_steps in ((.01, 1), (.1, 2), (0., 4)):
        method = regression.policy_method()
        method.args.mamba_target_kl = target_kl
        steps = []

        def overshoot(optimizer, loss, parameters):
            steps.append(float(loss.detach()))
            with torch.no_grad():
                method.actor.logits[0].add_(1.)

        method._step = overshoot
        record = regression.record(method, 32, 0)
        method._ppo_update([record], torch.ones(1))
        with torch.no_grad():
            log_ratio = (method.actor.routing_distribution(record['latent'], None, record['mask'])
                .log_prob(record['action']) - record['old_log_prob'])
            final_kl = float((log_ratio.exp() - 1. - log_ratio).mean())
        observations['kl_stop_' + str(target_kl)] = dict(target_kl=target_kl,
            optimizer_steps=len(steps), expected_optimizer_steps=expected_steps,
            optimizer_step_losses=steps, final_logits=method.actor.logits.detach().tolist(),
            final_sampled_approximate_kl=final_kl,
            passes=len(steps) == expected_steps,
            intervention='Each optimizer callback adds 1 to the selected action logit.')

    method = regression.policy_method()
    method.actor_optimizer = torch.optim.Adam(method.actor.parameters(), lr=.1)
    method._step = mamba.MambaMethod._step.__get__(method)
    initial = regression.record(method, 1, 0)
    loss = -method.actor.routing_distribution(initial['latent'], None, None).log_prob(initial['action']).mean()
    method._step(method.actor_optimizer, loss, list(method.actor.parameters()))
    actor_before = method.actor.logits.detach().clone()
    optimizer_before = deepcopy(method.actor_optimizer.state_dict())
    result = method._ppo_update([regression.record(method, 32, 0, forced=True)], torch.ones(1))
    optimizer_after = method.actor_optimizer.state_dict()
    delta = float((method.actor.logits.detach() - actor_before).abs().max())
    step_delta = float(optimizer_after['state'][0]['step'] - optimizer_before['state'][0]['step'])
    observations['forced_only_primed_adam'] = dict(
        actor_logits_before=actor_before.tolist(), actor_logits_after=method.actor.logits.detach().tolist(),
        max_abs_actor_parameter_delta=delta, adam_step_delta=step_delta,
        reported_actor_loss=float(result.detach()), passes=delta == 0. and step_delta == 0.)
    return observations


def run_child(source, variant):
    source = source.resolve()
    sys.path = [str(source), str(REPO / 'tests')] + [entry for entry in sys.path
        if entry and Path(entry).resolve() not in (REPO, FROZEN, REPO / 'tests')]
    import pytest
    import torch
    import urban_dreamer.learning.baselines.mamba as mamba
    import test_mamba_stability as regression
    import test_routing_mamba_baseline as helper

    torch.set_num_threads(1)
    assert Path(mamba.__file__).resolve() == source / 'urban_dreamer/learning/baselines/mamba.py'
    assert Path(regression.__file__).resolve() == TEST
    assert Path(helper.__file__).resolve() == REPO / 'tests/test_routing_mamba_baseline.py'
    directory = DESTINATION / variant
    directory.mkdir(parents=True, exist_ok=True)
    origin = dict(variant=variant, source=str(source), cwd=str(Path.cwd()),
        executable=sys.executable, sys_path=sys.path, mamba=fingerprint(mamba.__file__),
        helper=fingerprint(helper.__file__), regression_test=fingerprint(TEST),
        validation_script=fingerprint(__file__), torch_version=torch.__version__)
    (directory / 'imports.json').write_text(json.dumps(origin, indent=2) + '\n')
    observations = observe(regression, helper, mamba, pytest, torch)
    (directory / 'observations.json').write_text(json.dumps(observations, indent=2) + '\n')
    status = pytest.main([str(TEST), '-q', '-p', 'no:cacheprovider', '-k', SELECTION,
                          '--junitxml=' + str(directory / 'junit.xml')])
    package_files = {name: fingerprint(module.__file__)
        for name, module in sorted(sys.modules.items())
        if name.startswith('urban_dreamer') and getattr(module, '__file__', None)}
    assert all(Path(item['path']).is_relative_to(source / 'urban_dreamer')
               for item in package_files.values()), package_files
    (directory / 'package_imports.json').write_text(json.dumps(package_files, indent=2) + '\n')
    return int(status)


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
        command = [sys.executable, str(Path(__file__).resolve()), '--child-source',
                   str(source), '--variant', variant]
        result = subprocess.run(command, cwd=directory, env=env, capture_output=True,
                                text=True, timeout=180)
        (directory / 'pytest.log').write_text(result.stdout + result.stderr)
        cases = []
        xml_path = directory / 'junit.xml'
        if xml_path.exists():
            for case in ET.parse(xml_path).getroot().iter('testcase'):
                failure, error = case.find('failure'), case.find('error')
                status = 'failed' if failure is not None else 'error' if error is not None else 'passed'
                cases.append(dict(name=case.get('name'), status=status,
                    failure_message=failure.get('message') if failure is not None else None))
        variants[variant] = dict(source=str(source), returncode=result.returncode,
            command=command, imported=json.loads((directory / 'imports.json').read_text()),
            observations=json.loads((directory / 'observations.json').read_text()),
            counts={status: sum(case['status'] == status for case in cases)
                    for status in ('passed', 'failed', 'error')}, cases=cases)
    verified = (variants['frozen']['returncode'] == 1
        and variants['frozen']['counts'] == dict(passed=1, failed=7, error=0)
        and variants['current']['returncode'] == 0
        and variants['current']['counts'] == dict(passed=8, failed=0, error=0)
        and sum(item['passes'] for item in variants['frozen']['observations'].values()) == 1
        and all(item['passes'] for item in variants['current']['observations'].values()))
    report = dict(verified=verified, variants=variants,
        limitation=('Bounded mathematical defect regressions with controlled model heads, '
            'stub actor logits, injected KL overshoot and a primed real Adam optimizer. '
            'No SUMO training, convergence or throughput claim.'))
    (DESTINATION / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(verified=verified, summary=str(DESTINATION / 'summary.json'),
        results={name: data['counts'] for name, data in variants.items()}), indent=2))
    return 0 if verified else 1


if __name__ == '__main__':
    raise SystemExit(main())
