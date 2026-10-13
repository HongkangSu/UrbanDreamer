"""Prepare an isolated, provenance-checked MAPPO score-gauge continuation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import difflib
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

GAUGE = 'pre_head_legal_reference_v1'
POLICY_FILE = 'urban_dreamer/models/policy.py'
PPO_FILE = 'urban_dreamer/learning/baselines/ppo.py'
TRAINER_FILE = 'tools/train_toy_baseline.py'
OLD_SCORE = '        logits = self.policy_head(self.candidate_scorer(features)).squeeze(-1).masked_fill(~mask, -torch.inf)\n'
NEW_SCORE = '''        score_gauge = getattr(self, 'score_gauge', 'legacy')
        if score_gauge not in {'legacy', 'pre_head_legal_reference_v1'}:
            raise ValueError('unsupported actor score_gauge')
        hidden = self.candidate_scorer(features)
        if score_gauge == 'pre_head_legal_reference_v1':
            # Subtract before projection: subtracting large raw logits afterward
            # cannot recover the precision lost to their common offset.
            first = mask.to(torch.int64).argmax(dim=-1)
            reference = hidden[torch.arange(len(hidden), device=hidden.device), first]
            logits = torch.nn.functional.linear(
                hidden - reference[:, None, :], self.policy_head.weight, bias=None).squeeze(-1)
        else:
            logits = self.policy_head(hidden).squeeze(-1)
        logits = logits.masked_fill(~mask, -torch.inf)
'''
ACTOR_SETUP = '''        self.actor.score_gauge = getattr(args, 'score_gauge', 'legacy')
        if self.actor.score_gauge not in {'legacy', 'pre_head_legal_reference_v1'}:
            raise ValueError('unsupported PPO score_gauge')
'''
PROTOCOL_SETUP = '''        if self.actor.score_gauge != 'legacy':
            protocol.update(version=4, score_gauge=self.actor.score_gauge)
'''
CLI_SETUP = '''    parser.add_argument('--score-gauge', default='legacy',
                        choices=['legacy', 'pre_head_legal_reference_v1'],
                        help='Explicit MAPPO/IPPO score stabilization; changing this requires checkpoint protocol migration')
'''
CLI_VALIDATE = '''    if args.score_gauge != 'legacy' and args.method not in {'mappo', 'ippo'}:
        parser.error('nonlegacy score-gauge requires mappo or ippo')
'''


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def exact_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(exact_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and math.isnan(left):
        return math.isnan(right)
    return left == right


def migrate_checkpoint(payload):
    """Only the named protocol changes; all tensors/RNG/args remain exact."""
    if payload.get('method') not in {'mappo', 'ippo'}:
        raise ValueError('recovery requires a MAPPO/IPPO checkpoint')
    protocol = payload['method_state']['learning_protocol']
    if protocol.get('version') not in {2, 3} or 'score_gauge' in protocol:
        raise ValueError('only explicit legacy protocol versions 2/3 may migrate')
    migrated = deepcopy(payload)
    migrated['method_state']['learning_protocol'].update(version=4, score_gauge=GAUGE)
    restored = deepcopy(migrated)
    restored['method_state']['learning_protocol'] = deepcopy(protocol)
    if not exact_equal(payload, restored):
        raise ValueError('migration altered non-protocol checkpoint data')
    return migrated


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'frozen source anchor is missing or ambiguous: {old[:75]!r}')
    return text.replace(old, new, 1)


def overlay_source(files):
    """Apply only reviewed gauge hunks to OLD files, never current whole files."""
    changed = dict(files)
    changed[POLICY_FILE] = replace_once(files[POLICY_FILE], OLD_SCORE, NEW_SCORE)
    text = replace_once(files[PPO_FILE], '        self.actor = NextHopActor().to(self.device)\n',
                        '        self.actor = NextHopActor().to(self.device)\n' + ACTOR_SETUP)
    text = replace_once(text, '        return protocol\n', PROTOCOL_SETUP + '        return protocol\n')
    changed[PPO_FILE] = text
    text = replace_once(files[TRAINER_FILE], "    parser.add_argument('--ppo-minibatch-size', type=int, default=512)\n",
                        "    parser.add_argument('--ppo-minibatch-size', type=int, default=512)\n" + CLI_SETUP)
    changed[TRAINER_FILE] = replace_once(text, '    args = parser.parse_args(argv)\n',
                                       '    args = parser.parse_args(argv)\n' + CLI_VALIDATE)
    return changed


def continuation_argv(saved_args, source, output):
    args = dict(saved_args)
    args.update(config=str(source / 'configs' / Path(saved_args['config']).name),
                output_dir=str(output), resume=True, score_gauge=GAUGE)
    command = []
    for name, value in args.items():
        if value is None:
            continue
        if name == 'evaluation_argmax':
            command.append('--evaluation-argmax' if value else '--evaluation-sampled')
        elif name == 'global_reward_only':
            command.append('--global-reward-only' if value else '--no-global-reward-only')
        elif isinstance(value, bool):
            if value:
                command.append('--' + name.replace('_', '-'))
        else:
            command.extend(['--' + name.replace('_', '-'), str(value)])
    return command


# A save-boundary stop avoids changing --episodes and the entropy denominator.
WRAPPER = '''import hashlib, importlib.util, json, os, pathlib, subprocess, sys
import torch
root = pathlib.Path(__file__).resolve().parent
plan = json.loads((root / 'launch.json').read_text())
if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
    raise RuntimeError('verification must use physical GPU1 only')
free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '--id=1'], text=True).strip())
if free < 4096:
    raise RuntimeError('physical GPU1 has less than 4096 MiB free')
torch.set_num_threads(1)
torch.cuda.set_per_process_memory_fraction(.025, 0)
source = root / 'source'
for name, expected in json.loads((root / 'source_manifest.json').read_text()).items():
    if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('source integrity mismatch: ' + name)
sys.path.insert(0, str(source))
spec = importlib.util.spec_from_file_location('bounded_frozen_baseline', source / 'tools/train_toy_baseline.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
from urban_dreamer.learning.baselines.ppo import RoutingPpoMethod
if not pathlib.Path(sys.modules[RoutingPpoMethod.__module__].__file__).resolve().is_relative_to(source):
    raise RuntimeError('baseline module escaped frozen recovery source')
initial = torch.load(root / 'snapshot/latest.pt', map_location='cpu', weights_only=False)
if hashlib.sha256((root / 'run/latest.pt').read_bytes()).hexdigest() != plan['initial_checkpoint_sha256']:
    raise RuntimeError('prepared continuation checkpoint changed before launch')
passed = {}
original_update = RoutingPpoMethod.on_episode

def monitored_update(self, rows, windows, *, episode):
    result = original_update(self, rows, windows, episode=episode)
    passed.update(episode=episode, decisions=len(rows), windows=len(windows), strict_behavior_guard_passed=True,
                  update_metrics=result)
    return result
RoutingPpoMethod.on_episode = monitored_update
original_save = trainer.save
class BoundaryReached(BaseException):
    pass

def save_then_stop(runner, method, args, path, episode, best):
    original_save(runner, method, args, path, episode, best)
    if pathlib.Path(path).name == 'latest.pt' and episode == plan['stop_after_episode']:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        actor_changes = [name for name, value in saved['method_state']['actor'].items()
                         if not torch.equal(value, initial['method_state']['actor'][name])]
        critic_changes = [name for name, value in saved['method_state']['critic'].items()
                          if not torch.equal(value, initial['method_state']['critic'][name])]
        if saved['episode'] != episode or passed.get('episode') != episode or not actor_changes:
            raise RuntimeError('durable checkpoint/update verification failed')
        if saved['args']['episodes'] != initial['args']['episodes']:
            raise RuntimeError('entropy episode budget changed')
        archive = pathlib.Path(path).parent / 'episodes/train' / ('episode_%06d.npz' % episode)
        import numpy as np
        with np.load(archive, allow_pickle=False) as bank:
            summary = json.loads(bank['summary_json'].tobytes())
        metadata = summary['metadata']
        if metadata['demand_sha256'] != plan['expected_demand_sha256']:
            raise RuntimeError('demand realization changed')
        result = dict(status='completed_one_episode', **passed, actor_changed_parameters=actor_changes,
                      critic_changed_parameters=critic_changes, checkpoint_sha256=hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest(),
                      checkpoint_episode=episode, planned_episode_budget=saved['args']['episodes'], metadata=metadata,
                      stopped_at='after_durable_latest_checkpoint_before_next_episode', pid=os.getpid())
        (root / 'verification.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(status=result['status'], episode=episode, decisions=passed['decisions'],
                             actor_changed=len(actor_changes), critic_changed=len(critic_changes))), flush=True)
        raise BoundaryReached
trainer.save = save_then_stop
(root / 'process.json').write_text(json.dumps(dict(pid=os.getpid(), source=str(source), status='running'), indent=2))
try:
    trainer.main(plan['argv'])
except BoundaryReached:
    (root / 'process.json').write_text(json.dumps(dict(pid=os.getpid(), status='completed_one_episode'), indent=2))
except BaseException as error:
    (root / 'process.json').write_text(json.dumps(dict(pid=os.getpid(), status='failed', error=repr(error)), indent=2))
    raise
else:
    raise RuntimeError('expected bounded checkpoint boundary was not reached')
'''


def prepare(old_source, old_output, destination):
    old_source, old_output, destination = map(lambda p: Path(p).resolve(), (old_source, old_output, destination))
    if destination.exists():
        raise ValueError('refusing to overwrite an existing recovery directory')
    payload = torch.load(old_output / 'latest.pt', map_location='cpu', weights_only=False)
    migrated = migrate_checkpoint(payload)
    original_files = {name: (old_source / name).read_text() for name in (POLICY_FILE, PPO_FILE, TRAINER_FILE)}
    overlay = overlay_source(original_files)
    config = Path(payload['args']['config'])
    if config.parent != old_source / 'configs':
        raise ValueError('checkpoint config is outside original frozen source configs')
    next_episode = int(payload['episode']) + 1
    if payload['args'].get('checkpoint_every', 1) != 1:
        raise ValueError('one-episode verification requires original checkpoint_every=1')
    failed_archive = old_output / 'episodes/train' / f'episode_{next_episode:06d}.npz'
    with np.load(failed_archive, allow_pickle=False) as bank:
        old_summary = json.loads(bank['summary_json'].tobytes())
    if old_summary['metadata']['episode'] != next_episode:
        raise ValueError('failed archive episode mismatch')
    original_hashes = {str(old_output / name): digest(old_output / name) for name in ('latest.pt', 'metrics.jsonl')}
    original_hashes[str(failed_archive)] = digest(failed_archive)
    destination.mkdir(parents=True)
    source = destination / 'source'
    shutil.copytree(old_source, source, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    for name, text in overlay.items():
        (source / name).write_text(text)
    diff = ''.join(''.join(difflib.unified_diff(original_files[name].splitlines(True), overlay[name].splitlines(True),
                                              fromfile='old/' + name, tofile='new/' + name)) for name in overlay)
    (destination / 'gauge_only_source.patch').write_text(diff)
    manifest = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*')) if p.is_file()}
    original_manifest = {name: digest(old_source / name) for name in manifest}
    changed_files = [name for name in manifest if manifest[name] != original_manifest[name]]
    if sorted(changed_files) != sorted(original_files):
        raise ValueError('recovery source changed outside the three reviewed gauge files')
    (destination / 'original_source_manifest.json').write_text(json.dumps(original_manifest, indent=2))
    (destination / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    snapshot = destination / 'snapshot'
    snapshot.mkdir()
    torch.save(migrated, snapshot / 'latest.pt')
    output = destination / 'run'
    output.mkdir()
    shutil.copy2(snapshot / 'latest.pt', output / 'latest.pt')
    for name in ('best.pt',):
        if (old_output / name).exists():
            original = torch.load(old_output / name, map_location='cpu', weights_only=False)
            torch.save(migrate_checkpoint(original), output / name)
    shutil.copy2(old_output / 'metrics.jsonl', output / 'metrics.jsonl')
    if (old_output / 'protocol.json').exists():
        protocol = json.loads((old_output / 'protocol.json').read_text())
        protocol['learning_protocol'] = migrated['method_state']['learning_protocol']
        protocol['recovery_ancestry'] = str(old_output)
        (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    restored = torch.load(snapshot / 'latest.pt', map_location='cpu', weights_only=False)
    if not exact_equal(migrated, restored):
        raise ValueError('serialized migrated checkpoint did not preserve exact data')
    if (source / 'configs' / config.name).read_bytes() != config.read_bytes():
        raise ValueError('copied config bytes changed')
    if any(digest(path) != sha for path, sha in original_hashes.items()):
        raise ValueError('original runtime artifacts changed during preparation')
    report = dict(status='prepared', old_source=str(old_source), old_output=str(old_output), source_manifest_file='source_manifest.json',
                  checkpoint_original_sha256=original_hashes[str(old_output / 'latest.pt')],
                  checkpoint_migrated_sha256=digest(snapshot / 'latest.pt'), original_artifact_hashes=original_hashes,
                  only_checkpoint_change='method_state.learning_protocol.version=4 and score_gauge=' + GAUGE,
                  original_protocol=payload['method_state']['learning_protocol'], source_changed_files=changed_files,
                  migrated_protocol=migrated['method_state']['learning_protocol'], preservation_verified=True,
                  restored_episode=payload['episode'], planned_episode_budget=payload['args']['episodes'],
                  config_sha256=digest(config), failed_archive_preserved=str(failed_archive))
    (destination / 'migration_manifest.json').write_text(json.dumps(report, indent=2))
    plan = dict(argv=continuation_argv(payload['args'], source, output), stop_after_episode=next_episode,
                initial_checkpoint_sha256=digest(output / 'latest.pt'),
                expected_demand_sha256=old_summary['metadata']['demand_sha256'], physical_gpu=1,
                gpu_memory_fraction=.025, minimum_free_mib=4096, cpu_threads=1)
    (destination / 'launch.json').write_text(json.dumps(plan, indent=2))
    (destination / 'verify_one_episode.py').write_text(WRAPPER)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-source', type=Path, required=True)
    parser.add_argument('--old-output', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    print(json.dumps(prepare(args.old_source, args.old_output, args.output_dir), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
