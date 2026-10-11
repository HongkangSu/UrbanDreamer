"""Install an explicitly authorized calibrated decoder while preserving online history."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.migrate_imagination_budget import (_assert_equal, _read_frozen, _copy_history,
    _validate_online, imagination_budget)
from tools.run_complete_pretrain import export_run_config
from tools.run_online_pilot import atomic_json, sha256_file
from urban_dreamer.learning.codec_training import make_decoder_optimizer
from urban_dreamer.models.world_model import FlowProjector
from urban_dreamer.workflow.checkpoint import CheckpointPhase, latest_checkpoint_path, load_checkpoint
from urban_dreamer.workflow.config import load_config, _semantic_fingerprint

KIND = 'calibrated_decoder_continuation'
CODEC_CHANGES = dict(decoder_pretrain_enabled=True, decoder_online_enabled=True,
    decoder_online_learning_rate=1e-5, decoder_allow_unready=True,
    occupied_speed_weight=1., slow_speed_weight=1.)


def validate_transfer_config(source, target):
    before, after = source.resolved_snapshot(), target.resolved_snapshot()
    for snapshot in (before, after):
        for key in CODEC_CHANGES:
            snapshot['codec'].pop(key)
        for key in ('output_dir', 'device'):
            snapshot['run'].pop(key)
    if before != after or any(getattr(target.codec, k) != v for k, v in CODEC_CHANGES.items()):
        raise ValueError('decoder continuation may change only authorized decoder settings, output and device')


def _initial_state(source_state, calibration, config):
    state = copy.deepcopy(source_state)
    world = state['models']['world_model']
    source_codec = {k.removeprefix('flow_projector.'): v for k, v in world.items()
                    if k.startswith('flow_projector.')}
    candidate = calibration['codec']
    if source_codec.keys() != candidate.keys():
        raise ValueError('calibration codec tensor names differ from source')
    changed = 0
    for key, value in candidate.items():
        if value.shape != source_codec[key].shape or value.dtype != source_codec[key].dtype:
            raise ValueError(f'calibration codec tensor shape or dtype differs: {key}')
        if not torch.isfinite(value).all():
            raise ValueError(f'calibration codec tensor is nonfinite: {key}')
        if key.startswith('decoder_'):
            world['flow_projector.' + key] = value.clone()
            changed += 1
        elif not torch.equal(value, source_codec[key]):
            raise ValueError(f'calibration changed nondecoder codec tensor: {key}')
    if not changed or calibration.get('updates', 0) <= 0:
        raise ValueError('calibration needs trained decoder tensors and positive updates')
    # Construction is independent of city size; preserve this process RNG too.
    with torch.random.fork_rng(devices=[]):
        projector = FlowProjector(adjacency=torch.zeros(1, 1), positive_speed=config.codec.speed_positive)
    for name, lr in (('decoder_pretrain', config.codec.decoder_pretrain_learning_rate),
                     ('decoder_online', config.codec.decoder_online_learning_rate)):
        if name in state['optimizers']:
            raise ValueError(f'source already contains {name}; refusing implicit optimizer reset')
        state['optimizers'][name] = make_decoder_optimizer(projector, lr=lr).state_dict()
        for group in state['optimizers'][name]['param_groups']:
            group['betas'] = list(group['betas'])
        state['metrics'][name + '_updates'] = 0.
    state['metrics']['codec_ready'] = 0.
    for key in list(state['metrics']):
        if key.startswith(('codec_validation_', 'codec_hybrid_validation_',
                           'decoder_pretrain_authorized_')):
            del state['metrics'][key]
    return state


def rewrite_protocol(source_protocol, config, manifest):
    protocol = copy.deepcopy(source_protocol)
    protocol.update(device=config.run.device, config_semantics_unchanged=False,
        runtime_config_fingerprint=config.semantic_fingerprint,
        calibrated_decoder_migration=copy.deepcopy(manifest))
    return protocol


def validate_decoder_continuation(output_dir, config, input_hash, pretrained_source_hash):
    output = Path(output_dir).resolve()
    manifest = json.loads((output / 'decoder_migration.json').read_text())
    if (manifest['kind'] != KIND or manifest['target_output'] != str(output) or
        manifest['target_config_fingerprint'] != config.semantic_fingerprint or manifest['input_hash'] != input_hash):
        raise ValueError('decoder continuation target identity differs')
    for filename, digest in manifest['frozen_artifacts'].items():
        if sha256_file(output / filename) != digest:
            raise ValueError(f'decoder continuation frozen {filename} hash differs')
    source_config = load_config(output / 'source_config.toml')
    validate_transfer_config(source_config, config)
    if (source_config.semantic_fingerprint != manifest['source_config_fingerprint'] or
        imagination_budget(source_config) != manifest['budgets'] or imagination_budget(config) != manifest['budgets']):
        raise ValueError('decoder continuation source fingerprint or budgets differ')
    protocol = json.loads((output / 'source_protocol.json').read_text())
    if protocol['source']['sha256'] != pretrained_source_hash or manifest['pretrained_source_sha256'] != pretrained_source_hash:
        raise ValueError('decoder continuation pretrained ancestry differs')
    source = _read_frozen(output / 'source_checkpoint.pt', source_config, input_hash)
    initial = _read_frozen(output / 'continuation_checkpoint.pt', config, input_hash)
    _validate_online(source, source_config)
    progress = _validate_online(initial, config)
    if source.next_unit != initial.next_unit or source.next_unit != manifest['global_completed_episodes']:
        raise ValueError('decoder continuation changed global episode cursor')
    calibration = torch.load(output / 'calibration_checkpoint.pt', map_location='cpu', weights_only=False)
    if manifest['calibration_updates'] != calibration['updates'] or manifest['calibration_status'] != calibration['status']:
        raise ValueError('decoder continuation calibration metadata differs')
    _assert_equal(_initial_state(source.training_state, calibration, config), initial.training_state)
    if json.loads((output / 'continuation_pilot_progress.json').read_text()) != progress:
        raise ValueError('decoder continuation progress differs')
    if json.loads((output / 'continuation_pilot_protocol.json').read_text()) != rewrite_protocol(protocol, config, manifest):
        raise ValueError('decoder continuation rewritten protocol differs')
    latest = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                             expected_input_hash=input_hash)
    if latest.phase is CheckpointPhase.COMPLETED:
        state = latest.training_state
        if (latest.next_unit != 0 or len(state['metrics'].get('online_pilot_progress', {}).get('online', [])) != config.run.online_training_episodes
            or state['metrics'].get('online_episodes_completed') != config.run.online_training_episodes
            or state['next_environment_episode'] != config.run.online_training_episodes):
            raise ValueError('decoder continuation completed counters differ')
    else:
        _validate_online(latest, config, minimum_cursor=source.next_unit)
    return manifest


def migrate(source_dir, calibration_path, output_dir, *, device=None):
    source_dir, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output.exists() or output == source_dir:
        raise ValueError('decoder continuation requires a new absent output directory')
    source_config = load_config(source_dir / 'run_config.toml')
    config = replace(source_config, codec=replace(source_config.codec, **CODEC_CHANGES),
        run=replace(source_config.run, output_dir=output, device=device or source_config.run.device))
    validate_transfer_config(source_config, config)
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{k: getattr(config, k)
        for k in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=output.parent))
    try:
        for source, filename in ((latest_checkpoint_path(source_dir), 'source_checkpoint.pt'),
            (source_dir / 'run_config.toml', 'source_config.toml'),
            (source_dir / 'pilot_protocol.json', 'source_protocol.json'),
            (Path(calibration_path), 'calibration_checkpoint.pt')):
            shutil.copy2(source, temporary / filename)
        if (load_config(temporary / 'source_config.toml').semantic_fingerprint != source_config.semantic_fingerprint
            or sha256_file(source_dir / 'run_config.toml') != sha256_file(temporary / 'source_config.toml')):
            raise ValueError('decoder continuation source configuration changed during freezing')
        raw = torch.load(temporary / 'source_checkpoint.pt', map_location='cpu', weights_only=True)
        source = _read_frozen(temporary / 'source_checkpoint.pt', source_config, raw['input_hash'])
        progress = _validate_online(source, source_config)
        protocol = json.loads((temporary / 'source_protocol.json').read_text())
        budgets = dict(protocol.get('budgets', {}))
        budgets.setdefault('imagination_time_budget_seconds', 0.)
        if (budgets != imagination_budget(source_config) or protocol.get('online_episode_budget') != config.run.online_training_episodes
            or protocol['source'].get('input_hash', source.input_hash) != source.input_hash):
            raise ValueError('decoder continuation source protocol budgets or identity differ')
        calibration = torch.load(temporary / 'calibration_checkpoint.pt', map_location='cpu', weights_only=False)
        raw['training_state'] = _initial_state(source.training_state, calibration, config)
        raw['config_fingerprint'] = config.semantic_fingerprint
        export_run_config(config, temporary / 'run_config.toml')
        torch.save(raw, temporary / 'continuation_checkpoint.pt')
        latest_checkpoint_path(temporary).parent.mkdir(parents=True)
        shutil.copy2(temporary / 'continuation_checkpoint.pt', latest_checkpoint_path(temporary))
        copied = _copy_history(source_dir, temporary, online_cursor=source.next_unit)
        atomic_json(temporary / 'continuation_pilot_progress.json', progress)
        names = ('source_checkpoint.pt', 'source_config.toml', 'source_protocol.json',
                 'calibration_checkpoint.pt', 'continuation_checkpoint.pt', 'continuation_pilot_progress.json')
        hashes = {name: sha256_file(temporary / name) for name in names}
        hashes.update(copied)
        manifest = dict(schema_version=1, kind=KIND, source_output=str(source_dir), target_output=str(output),
            source_config_fingerprint=source_config.semantic_fingerprint, target_config_fingerprint=config.semantic_fingerprint,
            input_hash=source.input_hash, pretrained_source_sha256=protocol['source']['sha256'],
            global_completed_episodes=source.next_unit, budgets=imagination_budget(config),
            source_checkpoint_sha256=hashes['source_checkpoint.pt'],
            target_initial_checkpoint_sha256=hashes['continuation_checkpoint.pt'],
            calibration_sha256=hashes['calibration_checkpoint.pt'], calibration_updates=calibration['updates'],
            calibration_status=calibration['status'], explicitly_authorized_unready=True,
            optimizer_initialization='fresh Adam decoder_pretrain and decoder_online; all existing optimizers preserved',
            frozen_artifacts=hashes)
        atomic_json(temporary / 'decoder_migration.json', manifest)
        new_protocol = rewrite_protocol(protocol, config, manifest)
        atomic_json(temporary / 'continuation_pilot_protocol.json', new_protocol)
        atomic_json(temporary / 'pilot_protocol.json', new_protocol)
        atomic_json(temporary / 'pilot_progress.json', copy.deepcopy(progress))
        temporary.rename(output)
        validate_decoder_continuation(output, config, source.input_hash, protocol['source']['sha256'])
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--calibration-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device')
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(migrate(args.source_dir, args.calibration_path, args.output_dir, device=args.device), indent=2))


if __name__ == '__main__':
    main()
