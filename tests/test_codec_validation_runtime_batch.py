from dataclasses import replace

import pytest

from tools.run_complete_pretrain import MonitoredTrainer
from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.workflow.config import load_config


@pytest.mark.parametrize('fails', [False, True])
def test_validation_batch_override_preserves_training_config(monkeypatch, fails):
    trainer = object.__new__(MonitoredTrainer)
    original = load_config('configs/toy_cross_bottleneck.toml')
    original = replace(original, codec=replace(original.codec, batch_size=4))
    trainer.config = original
    trainer._codec_validation_batch_size = 32

    def validate(self):
        assert self.config.codec.batch_size == 32
        assert self.config.semantic_fingerprint == original.semantic_fingerprint
        assert self.config.resources == original.resources
        if fails:
            raise RuntimeError('validation failed')
        return {'codec_ready': True}

    monkeypatch.setattr(MainlineTrainer, '_validate_codec', validate)
    if fails:
        with pytest.raises(RuntimeError, match='validation failed'):
            trainer._validate_codec()
    else:
        assert trainer._validate_codec() == {'codec_ready': True}
    assert trainer.config is original
