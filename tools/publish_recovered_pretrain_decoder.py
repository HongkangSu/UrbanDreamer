"""Verify and atomically publish a decoder candidate without changing its frozen ancestry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tools.calibrate_frozen_decoder import file_hash
from tools.migrate_mainline_prior import _assert_equal
from tools.migrate_pretrain_single_step import _make_trainer
from tools.probe_slow_codec import atomic_json
from tools.recover_pretrain_decoder import candidate_training_state
from tools.run_city_pretrain_supplement import argument, process_table
from urban_dreamer.workflow.config import load_config


def verify_candidate(source, candidate, proof, *, protected_hashes, candidate_sha256):
    if (proof.get('status') != 'candidate_ready' or proof.get('proof', {}).get('codec_ready') is not True
            or proof.get('candidate_sha256') != candidate_sha256
            or proof.get('protected_sha256_before') != protected_hashes
            or proof.get('protected_sha256_after') != protected_hashes):
        raise ValueError('decoder candidate proof or frozen source hashes differ')
    _assert_equal({key: value for key, value in source.items() if key != 'training_state'},
                  {key: value for key, value in candidate.items() if key != 'training_state'})
    expected = candidate_training_state(source['training_state'], candidate['training_state'],
                                        updates=proof['updates'], accepted=True)
    _assert_equal(expected, candidate['training_state'])
    for prefix, proof_prefix in (('codec_validation_', ''), ('codec_hybrid_validation_', 'hybrid_')):
        for key, value in candidate['training_state']['metrics'].items():
            if key.startswith(prefix + 'codec_'):
                measured_key = proof_prefix + key.removeprefix(prefix)
                if measured_key in proof['proof']:
                    _assert_equal(value, proof['proof'][measured_key])


def assert_no_training_process(run_dir):
    for row in process_table().values():
        argv = row['argv']
        if any(Path(word).name in {'run_complete_pretrain.py', 'train_codec_ready.py',
                                   'migrate_pretrain_corpus.py'} for word in argv):
            if argument(argv, '--output-dir') == str(run_dir):
                raise ValueError(f'cannot publish while training process {row["pid"]} owns this run')


def publish(run_dir, recovery_dir):
    run_dir, recovery_dir = Path(run_dir).resolve(), Path(recovery_dir).resolve()
    proof = json.loads((recovery_dir / 'recovery_proof.json').read_text())
    candidate_path = recovery_dir / 'candidate_checkpoint.pt'
    checkpoint = run_dir / 'checkpoints/latest.pt'
    if proof.get('source_checkpoint') != str(checkpoint):
        raise ValueError('decoder candidate belongs to another source run')
    config = load_config(run_dir / 'run_config.toml')
    protected = [checkpoint, run_dir / 'run_config.toml', config.data.pretrain_cache_manifest]
    hashes = {str(path): file_hash(path) for path in protected}
    source = torch.load(checkpoint, map_location='cpu', weights_only=True)
    candidate = torch.load(candidate_path, map_location='cpu', weights_only=True)
    candidate_hash = file_hash(candidate_path)
    verify_candidate(source, candidate, proof, protected_hashes=hashes, candidate_sha256=candidate_hash)
    assert_no_training_process(run_dir)
    trainer, input_hash = _make_trainer(config)
    try:
        if candidate['config_fingerprint'] != config.semantic_fingerprint or candidate['input_hash'] != input_hash:
            raise ValueError('decoder candidate prepared inputs or configuration differ')
        trainer.load_state_dict(candidate['training_state'])
        if not trainer.codec_ready():
            raise ValueError('real MainlineTrainer codec_ready rejects the recovery candidate')
    finally:
        trainer.close()
    with (recovery_dir / 'publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_no_training_process(run_dir)
        if {str(path): file_hash(path) for path in protected} != hashes:
            raise ValueError('frozen source changed during decoder publication validation')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = checkpoint.parent / ('decoder_recovery_backup_' + stamp + '.pt')
        shutil.copy2(checkpoint, backup)
        temporary = checkpoint.with_name('latest.decoder_recovery.tmp')
        shutil.copy2(candidate_path, temporary)
        if file_hash(temporary) != candidate_hash or file_hash(backup) != hashes[str(checkpoint)]:
            raise ValueError('decoder publication temporary or backup hash differs')
        temporary.replace(checkpoint)
        published = dict(status='published', source_checkpoint=str(checkpoint), backup=str(backup),
            protected_sha256_before=hashes, published_checkpoint_sha256=file_hash(checkpoint),
            candidate_sha256=candidate_hash, real_mainline_codec_ready=True, updates=proof['updates'],
            verification='full heldout gate; all non-decoder tensors/optimizers and source RNG/replay/counters preserved')
        if published['published_checkpoint_sha256'] != candidate_hash:
            raise ValueError('published decoder checkpoint hash differs')
        atomic_json(recovery_dir / 'publication.json', published)
    return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--recovery-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.run_dir, args.recovery_dir), indent=2), flush=True)


if __name__ == '__main__':
    main()
