"""Reproduce PPO repair regressions against isolated frozen/current imports."""
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
DESTINATION = REPO / 'outputs/baseline_validation_20261015/ppo_regressions'
TEST = REPO / 'tests/test_ppo_stability.py'


def fingerprint(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run_child(source, variant):
    # Import production modules only after removing cwd/PYTHONPATH ambiguity.
    source = source.resolve()
    sys.path = [str(source), str(REPO / 'tests')] + [entry for entry in sys.path
        if entry and Path(entry).resolve() not in (REPO, FROZEN, REPO / 'tests')]
    import pytest
    import torch
    import urban_dreamer.learning.baselines.ppo as ppo
    import test_routing_ppo_baselines as helper

    torch.set_num_threads(1)
    assert Path(ppo.__file__).resolve() == source / 'urban_dreamer/learning/baselines/ppo.py'
    assert Path(helper.__file__).resolve() == REPO / 'tests/test_routing_ppo_baselines.py'
    directory = DESTINATION / variant
    directory.mkdir(parents=True, exist_ok=True)
    origin = dict(variant=variant, source=str(source), cwd=str(Path.cwd()),
        executable=sys.executable, sys_path=sys.path, ppo=fingerprint(ppo.__file__),
        helper=fingerprint(helper.__file__), regression_test=fingerprint(TEST),
        torch_version=torch.__version__)
    (directory / 'imports.json').write_text(json.dumps(origin, indent=2) + '\n')

    observations = {}
    for cls in (ppo.IppoMethod, ppo.MappoMethod):
        model = helper.method(cls)
        model.minibatch_size = 1
        rows, windows = helper.data(model)
        original_step = model.actor_optimizer.step

        def overshoot(*args, **kwargs):
            result = original_step(*args, **kwargs)
            with torch.no_grad():
                model.actor.policy_head.weight.mul_(10000.)
            return result

        model.actor_optimizer.step = overshoot
        kl_metrics = model.on_episode(rows, windows, episode=1)
        model = helper.method(cls)
        model.on_episode(*helper.data(model), episode=1)
        rows, windows = helper.data(model)
        for row in rows:
            row['mask'][:] = False
            row['mask'][row['action']] = True
            row['old_log_prob'] = 0.
        actor_before = deepcopy(model.actor.state_dict())
        optimizer_before = deepcopy(model.actor_optimizer.state_dict())
        forced_metrics = model.on_episode(rows, windows, episode=2)
        actor_delta = max(float((value - actor_before[name]).abs().max())
                          for name, value in model.actor.state_dict().items())
        step_deltas = [float(state['step'] - optimizer_before['state'][key]['step'])
                      for key, state in model.actor_optimizer.state_dict()['state'].items()]
        observations[cls.__name__] = dict(
            minibatch_kl=dict(optimizer_steps=kl_metrics['actor_optimizer_steps'],
                policy_kl=kl_metrics['policy_kl'], target_kl=model.target_kl,
                early_stopped=kl_metrics['kl_early_stopped']),
            forced_only=dict(actor_optimizer_steps=forced_metrics['actor_optimizer_steps'],
                critic_optimizer_steps=forced_metrics['critic_optimizer_steps'],
                actor_max_abs_parameter_delta=actor_delta,
                adam_step_deltas=sorted(set(step_deltas))))
    (directory / 'observations.json').write_text(json.dumps(observations, indent=2) + '\n')

    status = pytest.main([str(TEST), '-q', '-p', 'no:cacheprovider',
                          '--junitxml=' + str(directory / 'junit.xml')])
    package_files = {name: str(Path(module.__file__).resolve())
        for name, module in sorted(sys.modules.items())
        if name.startswith('urban_dreamer') and getattr(module, '__file__', None)}
    assert all(Path(path).is_relative_to(source / 'urban_dreamer')
               for path in package_files.values()), package_files
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
                failure = case.find('failure')
                error = case.find('error')
                state = 'failed' if failure is not None else 'error' if error is not None else 'passed'
                cases.append(dict(name=case.get('name'), status=state,
                    failure_message=failure.get('message') if failure is not None else None))
        variants[variant] = dict(source=str(source), returncode=result.returncode,
            command=command, imported=json.loads((directory / 'imports.json').read_text()),
            observations=json.loads((directory / 'observations.json').read_text()),
            counts={state: sum(case['status'] == state for case in cases)
                    for state in ('passed', 'failed', 'error')}, cases=cases)
    verified = (variants['frozen']['returncode'] == 1
        and variants['frozen']['counts'] == dict(passed=1, failed=6, error=0)
        and variants['current']['returncode'] == 0
        and variants['current']['counts'] == dict(passed=7, failed=0, error=0))
    report = dict(verified=verified, variants=variants,
        limitation='Bounded defect regressions with injected KL overshoot; no convergence or throughput claim.')
    (DESTINATION / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(verified=verified, summary=str(DESTINATION / 'summary.json'),
        results={name: data['counts'] for name, data in variants.items()}), indent=2))
    return 0 if verified else 1


if __name__ == '__main__':
    raise SystemExit(main())
