"""Paired fresh-run SUMO verification, isolated from production experiments."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/baseline_validation_20261015'
OLD = ROOT / 'outputs/city_baselines_current_to_dest_1000_20261013_180607/source'
PYTHON = '/home/wangwenrui/opt/miniconda3/envs/URB/bin/python'
METHODS = ('ippo', 'mappo', 'mamba', 'mbpo', 'dreamer')


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare():
    source = OUT / 'source_repaired'
    if (OUT / 'source_original').exists():
        return
    hashes = {}
    for directory in ('urban_dreamer', 'tools'):
        for path in sorted((ROOT / directory).rglob('*.py')):
            relative = path.relative_to(ROOT)
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = path.read_bytes()
            target.write_bytes(data)
            hashes[str(relative)] = hashlib.sha256(data).hexdigest()
    # Hold simulator/config/cache code fixed. Only restore the algorithm files
    # and shared actor update from the original experiment's frozen source.
    original = OUT / 'source_original'
    shutil.copytree(source, original, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('ppo', 'mamba', 'mbpo', 'dreamer'):
        relative = Path('urban_dreamer/learning/baselines') / (name + '.py')
        shutil.copy2(OLD / relative, original / relative)
    relative = Path('urban_dreamer/learning/baselines/common.py')
    common = (original / relative).read_text()
    old_common = (OLD / relative).read_text()
    begin = common.index('def policy_gradient_step(')
    end = common.index('\ndef save_checkpoint(', begin)
    old_begin = old_common.index('def policy_gradient_step(')
    old_end = old_common.index('\ndef save_checkpoint(', old_begin)
    (original / relative).write_text(common[:begin] + old_common[old_begin:old_end] + common[end:])
    text = (ROOT / 'configs/toy_cross_bottleneck.toml').read_text()
    text = text.replace('network_dir = "../networks/toy_cross_bottleneck"',
        'network_dir = ' + json.dumps(str(ROOT / 'networks/toy_cross_bottleneck')))
    # Bound exploratory failures while retaining the full 320-vehicle demand.
    text = re.sub(r'^world_model_intervals_per_episode = .*$',
        'world_model_intervals_per_episode = 30', text, flags=re.M)
    text = re.sub(r'^training_completion_buffer_seconds = .*$',
        'training_completion_buffer_seconds = 300', text, flags=re.M)
    text = re.sub(r'^evaluation_horizon_seconds = .*$',
        'evaluation_horizon_seconds = 1200', text, flags=re.M)
    (OUT / 'toy_validation.toml').write_text(text)
    write(OUT / 'source_hashes.json', hashes)
    write(OUT / 'plan.json', dict(original_algorithm_archive=str(OLD),
        old_source=str(original), repaired_source=str(source),
        held_fixed='identical simulator, config parser, route candidates/cache, observations and collection',
        methods=METHODS, training_seeds=[42, 43, 44], training_episodes=100,
        evaluation_episode_indices=[10042, 10043, 10044, 10045, 10046],
        evaluation_policies=['argmax', 'sampled_temperature_1'],
        demand='full toy demand, unchanged Poisson arrivals over 420 seconds',
        physics='meso/libsumo/event/current_to_dest, policy every 30 seconds',
        deadline_seconds=1200, entropy_horizon=1000,
        comparison='fresh initial weights, original defaults vs repaired defaults',
        limits='Toy screening and paired evidence; does not establish city convergence'))


def worker(options):
    import fcntl

    output = OUT / 'runs' / options.method / f'{options.variant}_seed{options.seed}'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX)
    completed_path = output / 'completed.json'
    if completed_path.exists() and json.loads(completed_path.read_text())['episodes'] == options.episodes:
        return
    source = OUT / ('source_original' if options.variant == 'original' else 'source_repaired')
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from urban_dreamer.learning.baselines import build_method, common
    from urban_dreamer.learning.baselines.collection_policy import CollectionPolicy
    from tools.train_toy_baseline import apply_episode_entropy

    torch.set_num_threads(1)
    metrics_path = output / 'metrics.jsonl'
    if metrics_path.exists():
        previous = json.loads((output / 'provenance.json').read_text())
        # The controller may reach a slow job already started independently.
        # Adopt that run rather than creating a second training process.
        while Path('/proc', str(previous['pid'])).exists() and not (output / 'completed.json').exists():
            time.sleep(2.)
        if (output / 'completed.json').exists():
            completed = json.loads((output / 'completed.json').read_text())
            if completed['episodes'] == options.episodes:
                return
        raise ValueError(f'refusing to overwrite {metrics_path}')
    args = argparse.Namespace(config=OUT / 'toy_validation.toml', output_dir=output,
        method=options.method, device='cpu', threads=1, seed=options.seed, episodes=options.episodes,
        vehicle_limit=0, actor_lr=2e-4, critic_lr=3e-4, session=None, backend=None, step_seconds=None,
        reward_objective='observed_mean', global_reward_only=True,
        score_gauge='pre_head_legal_reference_v1' if options.method in {'ippo', 'mappo'} else 'legacy',
        mamba_model_warmup_updates=100, mamba_min_model_frames=100, mamba_burn_in_windows=5,
        dreamer_imagination_device='auto', sp_prior_weight=0., sp_prior_temperature=30.)
    runner = common.EpisodeRunner(args)
    method = build_method(options.method, args, runner)
    collector = CollectionPolicy(method)
    method.collection_policy = collector
    actor_initial = [p.detach().clone() for p in method.actor.parameters()]
    provenance = dict(source=str(source), common_module=common.__file__,
        method_module=sys.modules[type(method).__module__].__file__, args=vars(args) | {
            'config': str(args.config), 'output_dir': str(args.output_dir)},
        config=runner.config.resolved_snapshot(), input_hashes=runner.prepared.input_hashes,
        learning_protocol=method.learning_protocol() if hasattr(method, 'learning_protocol') else {},
        pid=os.getpid(), started_at=time.time())
    write(output / 'provenance.json', provenance)

    def log(record):
        record = {k: v for k, v in record.items() if k not in {'choices', 'branch_choices'}}
        with metrics_path.open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')

    def evaluate(episode):
        for policy in ('argmax', 'sampled_temperature_1'):
            for offset in range(options.eval_seeds):
                seed = 10042 + offset
                with common.evaluation_rng(seed, method):
                    _, _, metrics = runner.run_episode(functools.partial(collector.select,
                        explore=False, sample_actions=policy != 'argmax', temperature=1.),
                        seed=seed, label=policy, on_demand=collector.begin_episode)
                log(metrics | dict(phase='evaluation', episode=episode, policy=policy,
                    evaluation_seed=seed))

    apply_episode_entropy(method, runner.config.method, completed_episodes=0, total_episodes=1000)
    if options.eval_seeds:
        evaluate(0)
    for episode in range(1, options.episodes + 1):
        entropy = apply_episode_entropy(method, runner.config.method,
            completed_episodes=episode - 1, total_episodes=1000)
        before = [p.detach().clone() for p in method.actor.parameters()]
        started = time.monotonic()
        rows, windows, metrics = runner.run_episode(functools.partial(collector.select, explore=True),
            seed=options.seed + episode, collect=getattr(method, 'needs_rows', True),
            record_windows=getattr(method, 'needs_window_records', False),
            row_flow=getattr(method, 'needs_row_flow', True), on_demand=collector.begin_episode)
        collected = time.monotonic()
        for row in rows:
            row['sp_prior_weight'] = 0.
        update = method.on_episode(rows, windows, episode=episode)
        elapsed = time.monotonic() - started
        squared = sum(float((p.detach() - b).square().sum())
            for p, b in zip(method.actor.parameters(), before))
        for name in ('actor', 'critic', 'global_critic', 'latent_critic', 'world_model', 'global_model', 'ensemble'):
            module = getattr(method, name, None)
            if module is not None and any(not bool(torch.isfinite(p).all()) for p in module.parameters()):
                raise RuntimeError(f'nonfinite {name} parameters at episode {episode}')
        effective = sum(np.count_nonzero(row['mask']) > 1 for row in rows)
        record = dict(phase='train', episode=episode, **metrics, **update, **entropy)
        record.update(total_seconds=elapsed, collection_measured_seconds=collected-started,
            learning_measured_seconds=elapsed-(collected-started), actor_delta_l2=squared**.5,
            actual_choice_count=int(effective))
        if hasattr(method, '_last_ppo_kl'):
            record['mamba_policy_kl'] = method._last_ppo_kl
        log(record)
        if episode % 10 == 0 or episode == options.episodes:
            print(json.dumps(dict(method=options.method, variant=options.variant, seed=options.seed,
                episode=episode, reward=metrics['global_reward_sum'],
                completion=metrics['completion_rate'], entropy=metrics['branch_entropy'],
                total_seconds=elapsed)), flush=True)
    if options.eval_seeds:
        evaluate(options.episodes)
    delta = sum(float((p.detach() - b).square().sum())
        for p, b in zip(method.actor.parameters(), actor_initial))**.5
    write(output / 'completed.json', dict(episodes=options.episodes, actor_total_delta_l2=delta,
        completed_at=time.time(), numerical_parameters_finite=True))


def launch(options):
    prepare()
    jobs = [(method, variant, seed) for seed in (42,43,44)
        for method in METHODS for variant in ('original','repaired')]
    started = time.time()
    write(OUT / 'launch.json', dict(pid=os.getpid(), started_at=started,
        concurrency=options.concurrency, episodes=options.episodes, jobs=jobs))
    def run(job):
        method, variant, seed = job
        output = OUT / 'runs' / method / f'{variant}_seed{seed}'
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'completed.json').exists():
            return dict(method=method,variant=variant,seed=seed,status='already_complete')
        command = [PYTHON, '-u', str(Path(__file__).resolve()), '--worker', '--method', method,
            '--variant', variant, '--seed', str(seed), '--episodes', str(options.episodes),
            '--eval-seeds', str(options.eval_seeds)]
        environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with (output / 'stdout.log').open('a') as stream:
            process = subprocess.Popen(command, cwd=output, env=environment,
                stdout=stream, stderr=subprocess.STDOUT)
            write(output / 'process.json', dict(pid=process.pid, command=command))
            code = process.wait()
        result = dict(method=method, variant=variant, seed=seed, exit_code=code,
            status='completed' if code == 0 else 'failed')
        print(json.dumps(result), flush=True)
        return result
    results = []
    with ThreadPoolExecutor(max_workers=options.concurrency) as executor:
        for future in as_completed([executor.submit(run, job) for job in jobs]):
            results.append(future.result())
            write(OUT / 'progress.json', dict(started_at=started, updated_at=time.time(), results=results))
    write(OUT / 'finished.json', dict(started_at=started, completed_at=time.time(), results=results))
    if any(row.get('exit_code', 0) for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--variant', choices=('original','repaired'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--eval-seeds', type=int, default=5)
    parser.add_argument('--concurrency', type=int, default=6)
    options = parser.parse_args()
    if options.prepare:
        prepare()
    elif options.worker:
        worker(options)
    else:
        launch(options)
