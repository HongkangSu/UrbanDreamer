"""Policy-changing continuation must preserve history without reusing old evaluation baselines."""
import copy
from types import SimpleNamespace

from tools.run_bounded_online_continuation import prepare_progress
from urban_dreamer.workflow.checkpoint import load_checkpoint
from test_urban_dreamer_training import make_trainer


def test_continuation_keeps_online_history_and_restarts_paired_baseline(tmp_path):
    trainer = make_trainer(tmp_path)
    history = dict(schema_version=1, online=[{'episode_index': 0}], before=[{'old': 'baseline'}],
                   after=[], milestones=[{'old': 'evaluation'}], phase='online_training')
    state = trainer.state_dict()
    state['metrics']['online_pilot_progress'] = copy.deepcopy(history)
    checkpoint = SimpleNamespace(training_state=state, next_unit=1, input_hash='inputs')
    protocol = {'migration_episode': 1}
    prepare_progress(tmp_path, checkpoint, protocol, trainer.config)
    loaded = load_checkpoint(output_dir=tmp_path, expected_config_fingerprint=trainer.config.semantic_fingerprint,
                             expected_input_hash='inputs')
    progress = loaded.training_state['metrics']['online_pilot_progress']
    assert loaded.next_unit == 1
    assert progress['online'] == history['online']
    assert progress['before'] == progress['after'] == progress['milestones'] == []
    assert progress['critic_prior_migration_episode'] == 1
    import json
    assert json.loads((tmp_path/'pre_migration_pilot_progress.json').read_text()) == history
    # Restart cannot erase baseline evaluation progress already committed.
    before = (tmp_path/'checkpoints/latest.pt').read_bytes()
    prepare_progress(tmp_path, checkpoint, protocol, trainer.config)
    assert (tmp_path/'checkpoints/latest.pt').read_bytes() == before


def test_fresh_pretrained_seed_starts_with_empty_online_history(tmp_path):
    trainer = make_trainer(tmp_path)
    checkpoint = SimpleNamespace(training_state=trainer.state_dict(), next_unit=0, input_hash='inputs')
    prepare_progress(tmp_path, checkpoint, {'migration_episode': 0}, trainer.config)
    loaded = load_checkpoint(output_dir=tmp_path, expected_config_fingerprint=trainer.config.semantic_fingerprint,
                             expected_input_hash='inputs')
    assert loaded.next_unit == 0
    progress = loaded.training_state['metrics']['online_pilot_progress']
    assert progress['online'] == progress['before'] == progress['milestones'] == []
    assert progress['critic_prior_migration_episode'] == 0
