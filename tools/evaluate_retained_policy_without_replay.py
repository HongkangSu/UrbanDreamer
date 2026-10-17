"""Evaluate retained policy with frozen trainer and no offline/replay restoration."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def restore_models_only(trainer, state):
    for name, weights in state['models'].items():
        getattr(trainer, name).load_state_dict(weights, strict=True)
    trainer.metrics = dict(state['metrics'])
    trainer._next_environment_episode = int(state['next_environment_episode'])
    trainer._set_online_prior()


@contextmanager
def forbid_old_data(paths):
    """Fail before opening/statting any registered old manifest or raw source."""
    forbidden = {str(Path(p).absolute()) for p in paths}
    attempts = []
    original_open, original_stat = Path.open, Path.stat
    def check(path):
        if str(path.absolute()) in forbidden:
            attempts.append(str(path))
            raise RuntimeError('Old training data access forbidden: ' + str(path))
    def guarded_open(path, *args, **kwargs):
        check(path)
        return original_open(path, *args, **kwargs)
    def guarded_stat(path, *args, **kwargs):
        check(path)
        return original_stat(path, *args, **kwargs)
    with patch.object(Path, 'open', guarded_open), patch.object(Path, 'stat', guarded_stat):
        yield attempts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'checkpoint', 'metadata', 'config', 'protected-sources', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--indices', type=int, nargs='+', default=[0])
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError('Use a fresh evaluation output directory')
    sys.path.insert(0, str(a.source.resolve()))
    import torch
    from urban_dreamer.learning.training import MainlineTrainer, evaluation_protocol, REWARD_CONTRACT
    from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
    from urban_dreamer.learning.offline_data import OfflineEpisodeDataset
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.traffic.topology import load_traffic_topology
    from urban_dreamer.workflow.config import load_config
    from urban_dreamer.workflow.pipeline import combined_input_hash
    torch.set_num_threads(2)
    original = load_config(a.config)
    metadata = json.loads(a.metadata.read_text())
    payload = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if sha(a.checkpoint) != metadata['checkpoint_sha256']:
        raise ValueError('retained checkpoint hash mismatch')
    if payload['config_fingerprint'] != original.semantic_fingerprint or payload['config_fingerprint'] != metadata['config_fingerprint']:
        raise ValueError('source config fingerprint mismatch')
    if payload['training_state']['reward_contract'] != REWARD_CONTRACT:
        raise ValueError('reward contract mismatch')
    config = replace(original, run=replace(original.run, output_dir=a.output.resolve(), device='cpu'),
        data=replace(original.data, archive_roots=(), validation_archive_roots=(), pretrain_cache_manifest=None, validation_cache_manifest=None))
    a.output.mkdir(parents=True)
    old_paths = [r['path'] for r in map(json.loads, a.protected_sources.read_text().splitlines())]
    old_paths += [str(v) for v in (original.data.pretrain_cache_manifest, original.data.validation_cache_manifest) if v]
    old_init = OfflineEpisodeDataset.__init__
    def empty_only(self, conf, *args, **kwargs):
        if conf.data.archive_roots or conf.data.pretrain_cache_manifest or conf.data.validation_archive_roots or conf.data.validation_cache_manifest:
            raise RuntimeError('offline data unexpectedly enabled')
        return old_init(self, conf, *args, **kwargs)
    protocol = evaluation_protocol(None, prior_weight=config.method.sp_prior_evaluation_weight, demand_seed=config.method.evaluation_demand_seed)
    if protocol != metadata['evaluation_protocol']:
        raise ValueError('evaluation protocol mismatch')
    result = dict(checkpoint_sha256=metadata['checkpoint_sha256'], original_config_fingerprint=original.semantic_fingerprint,
        evaluation_config_fingerprint=config.semantic_fingerprint, overrides={'run.device':'cpu','run.output_dir':str(a.output.resolve()),'data.archive_roots':[],'data.validation_archive_roots':[],'data.pretrain_cache_manifest':None,'data.validation_cache_manifest':None},
        source=str(a.source.resolve()), source_config=str(a.config.resolve()), evaluation_protocol=protocol, optimizer_restored=False, replay_restored=False, evaluations=[])
    (a.output/'evaluation_config_overrides.json').write_text(json.dumps(result,indent=2)+'\n')
    trainer = None
    try:
        with forbid_old_data(old_paths) as attempts, patch.object(CompactPretrainDataset, '__init__', side_effect=RuntimeError('compact dataset forbidden')), patch.object(OfflineEpisodeDataset, '__init__', empty_only):
            prepared = validate_prepared_inputs(config)
            if combined_input_hash(prepared) != payload['input_hash'] or payload['input_hash'] != metadata['input_hash']:
                raise ValueError('input hash mismatch')
            trainer = MainlineTrainer(config=config, prepared_inputs=prepared, topology=load_traffic_topology(prepared))
            restore_models_only(trainer, payload['training_state'])
            trainer.evaluation_temperature = None
            assert not trainer.offline_data.has_samples and trainer.codec_validation_data is None
            for index in a.indices:
                actual = trainer.evaluate_episode(index)
                demand = json.loads((a.output/'evaluation_demands'/f'episode_{index:04d}.json').read_text())
                expected = next(r for r in metadata['evaluations'] if r['episode_index']==index)
                keys=['evaluation_selection_score','evaluation_vehicle_count','evaluation_completion_rate','evaluation_cumulative_vehicle_seconds','evaluation_nonarrival_count','evaluation_sp_prior_schedule_episode']
                equal = {key: actual[key] == expected['metrics'][key] for key in keys}
                equal['demand_hash'] = demand['demand_hash'] == expected['demand_hash']
                result['evaluations'].append(dict(episode_index=index, actual=actual, demand_hash=demand['demand_hash'], exact_comparisons=equal))
                result.update(status='passed' if all(all(r['exact_comparisons'].values()) for r in result['evaluations']) else 'mismatch', forbidden_data_attempts=attempts)
                (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps({'index':index,'status':result['status'],'comparisons':equal}),flush=True)
            if result['status']!='passed':raise RuntimeError('evaluation mismatch')
    finally:
        if trainer is not None:trainer.close()

if __name__=='__main__':main()
