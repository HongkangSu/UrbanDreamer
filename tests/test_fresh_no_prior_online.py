from dataclasses import replace

import pytest

from tools.prepare_fresh_no_prior_online import target_config
from urban_dreamer.workflow.config import load_config


def test_restart_changes_only_output_and_removes_prior(tmp_path):
    source = load_config('configs/toy_cross_bottleneck.toml')
    source = replace(source,
        method=replace(source.method, critic_value_mode='bounded_cohort_v1',
                       sp_prior_weight=1., sp_prior_evaluation_weight=0., sp_prior_cost_mode='dynamic'),
        resources=replace(source.resources, imagination_rollouts_per_pass=8))
    target = target_config(source, tmp_path / 'new')
    before, after = source.resolved_snapshot(), target.resolved_snapshot()
    assert after['run']['output_dir'] == str((tmp_path / 'new').resolve())
    assert after['method']['sp_prior_weight'] == after['method']['sp_prior_evaluation_weight'] == 0.
    assert after['method']['sp_prior_cost_mode'] == 'free_flow'
    for values in (before, after):
        values['run'].pop('output_dir')
        for key in ('sp_prior_weight', 'sp_prior_evaluation_weight', 'sp_prior_cost_mode'):
            values['method'].pop(key)
    assert before == after
    assert target.resources.imagination_rollouts_per_pass == 8
    assert target.semantic_fingerprint


def test_restart_cannot_silently_change_critic_contract(tmp_path):
    with pytest.raises(ValueError, match='bounded'):
        target_config(load_config('configs/toy_cross_bottleneck.toml'), tmp_path / 'new')
