from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.component_policy_adapters import (FrozenActorPolicy, validate_contract, simulation_contract,
                                             snapshot_checkpoint, load_policy)
from urban_dreamer.learning.observations import CandidateObservations
from urban_dreamer.models.policy import NextHopActor
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.pipeline import combined_input_hash


@pytest.fixture
def observations():
    features = np.random.default_rng(7).random((2, 3, 10), dtype=np.float32)
    mask = np.asarray([[True, False, True], [False, True, True]])
    return CandidateObservations(features, mask, np.asarray([[2, -1, 4], [-1, 6, 7]]))


@pytest.mark.parametrize('method', ['urban_dreamer', 'mappo', 'ippo', 'dreamer'])
def test_production_actor_selection_is_masked_argmax_and_frozen_across_episodes(method, observations):
    torch.set_num_threads(2)
    actor = NextHopActor()
    actor.prior_weight = 5.
    policy = FrozenActorPolicy(actor, method, {})
    vehicles = [dict(vehicle_id='v0', current_edge=0), dict(vehicle_id='v1', current_edge=1)]
    policy.begin_episode(seed=610001)
    actual, _ = policy.select(vehicles, observations, None, 30., explore=False)
    with torch.no_grad():
        expected = actor.forward_logits(torch.from_numpy(observations.features),
                                        torch.from_numpy(observations.mask)).argmax(-1).tolist()
    assert actual == expected
    first = policy.end_episode()
    assert first['optimizer_updates'] == 0
    assert first['select_calls'] == 1
    policy.begin_episode(seed=610002)
    assert policy._selections == 0
    assert policy.select(vehicles, observations, None, 30., explore=False)[0] == expected
    assert policy.end_episode()['actor_state_sha256'] == first['actor_state_sha256']
    assert not actor.training and all(not p.requires_grad for p in actor.parameters())
    assert actor.prior_weight == 0


def test_snapshot_rejects_input_and_physics_mismatches():
    config = load_config('configs/ingolstadt_custom.toml')
    runner = SimpleNamespace(config=config, args=SimpleNamespace(forbid_revisit=False),
                             prepared=SimpleNamespace(input_hashes={'network': 'abc'}))
    provenance = dict(input_hashes=runner.prepared.input_hashes,
        combined_input_hash=combined_input_hash(runner.prepared), simulation=simulation_contract(config))
    validate_contract(provenance, runner)
    with pytest.raises(ValueError, match='network/input'):
        validate_contract({**provenance, 'input_hashes': {'network': 'other'}}, runner)
    changed = {**provenance['simulation'], 'vehicle_max_speed_mps': 100.}
    with pytest.raises(ValueError, match='physical/observation'):
        validate_contract({**provenance, 'simulation': changed}, runner)


def test_frozen_actor_detects_parameter_changes():
    torch.set_num_threads(2)
    policy = FrozenActorPolicy(NextHopActor(), 'urban_dreamer', {})
    policy.begin_episode(seed=610001)
    with torch.no_grad():
        next(policy.actor.parameters()).add_(.1)
    with pytest.raises(RuntimeError, match='weights'):
        policy.end_episode()


def test_mainline_snapshot_preserves_commit_and_rejects_tampered_actor(tmp_path):
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.workflow.checkpoint import save_checkpoint
    config_path = 'configs/ingolstadt_custom.toml'
    config = load_config(config_path)
    prepared = validate_prepared_inputs(config)
    actor = NextHopActor()
    state = dict(reward_contract='causal_cumulative_mean_global_v1', models={'actor': actor.state_dict()},
        optimizers={}, replay={}, rng={}, metrics={'online_episodes_completed': 7}, next_environment_episode=7)
    source = save_checkpoint(output_dir=tmp_path / 'producer',
        config_fingerprint=config.semantic_fingerprint, input_hash=combined_input_hash(prepared),
        phase='online_training', next_unit=7, training_state=state)
    output = tmp_path / 'snapshot'
    provenance = snapshot_checkpoint(source=source, config=config_path, output=output)
    assert provenance['episode']['committed_online_episodes'] == 7
    assert provenance['checkpoint_sha256']
    runner = SimpleNamespace(prepared=prepared, config=config, args=SimpleNamespace(forbid_revisit=False))
    loaded = load_policy(checkpoint=output / 'checkpoints/latest.pt', runner=runner)
    loaded.begin_episode(seed=610001)
    assert loaded.end_episode()['optimizer_updates'] == 0
    with (output / 'actor.pt').open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='actor artifact digest'):
        load_policy(checkpoint=output / 'checkpoints/latest.pt', runner=runner)


@pytest.mark.parametrize('method', ['mappo', 'ippo', 'dreamer'])
def test_baseline_snapshot_extracts_actual_checkpoint_actor_schema(tmp_path, method):
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.traffic.simulation_settings import physics_identity
    from urban_dreamer.learning.baselines.common import CHECKPOINT_FORMAT
    config_path = 'configs/ingolstadt_custom.toml'
    config = load_config(config_path)
    prepared = validate_prepared_inputs(config)
    actor = NextHopActor()
    source = tmp_path / 'baseline.pt'
    torch.save(dict(format=CHECKPOINT_FORMAT, method=method, episode=12,
        method_state={'actor': actor.state_dict()}, input_hashes=prepared.input_hashes,
        config_fingerprint=config.semantic_fingerprint, physics=physics_identity(config.simulation),
        args={'reward_objective': 'observed_mean', 'forbid_revisit': False}), source)
    output = tmp_path / 'snapshot'
    provenance = snapshot_checkpoint(source=source, config=config_path, output=output)
    assert provenance['method'] == method
    assert provenance['episode']['committed_online_episodes'] == 12
    runner = SimpleNamespace(prepared=prepared, config=config, args=SimpleNamespace(forbid_revisit=False))
    policy = load_policy(checkpoint=output / 'checkpoints/latest.pt', runner=runner)
    assert policy.method == method
    assert not policy.actor.training
