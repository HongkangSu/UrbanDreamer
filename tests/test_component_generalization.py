import json

import pytest

from tools.run_component_generalization import (
    FrozenDemand, freeze_json, normalize_metrics, scaled_demand, validate_seed_split,
)
from urban_dreamer.traffic.demand import PoissonDemand, demand_identity


def test_scale_changes_poisson_rates_without_changing_duration_or_od():
    base = PoissonDemand(((1, 3, 2.), (2, 4, 1.)), 30.)
    shifted = scaled_demand(base, 1.3, 'original')
    assert shifted.od_rates == ((1, 3, 2.6), (2, 4, 1.3))
    assert shifted.expected_vehicle_count == pytest.approx(117.)
    assert shifted.duration_seconds == 30
    assert base.expected_vehicle_count == 90


@pytest.mark.parametrize('profile', ['original', 'peaks', 'burst', 'od_stagger'])
def test_retiming_preserves_sampled_vehicle_population_and_pairing(profile):
    base = PoissonDemand(((1, 3, 2.), (2, 4, 1.)), 30.)
    shifted = scaled_demand(base, .7, profile)
    vehicles = shifted.sample(seed=42, episode_index=610001)
    original = scaled_demand(base, .7, 'original').sample(seed=42, episode_index=610001)
    population = lambda rows: sorted((v.vehicle_id, v.origin_index, v.destination_index) for v in rows)
    assert population(vehicles) == population(original)
    frozen = FrozenDemand(shifted, vehicles, 42, 610001)
    assert demand_identity(frozen.sample(seed=42, episode_index=610001)) == demand_identity(vehicles)
    with pytest.raises(ValueError, match='seed mismatch'):
        frozen.sample(seed=42, episode_index=710001)


def test_manifest_refuses_changed_demand_on_resume(tmp_path):
    path = tmp_path / 'demand.json'
    freeze_json(path, dict(seed=610001))
    freeze_json(path, dict(seed=610001))
    with pytest.raises(ValueError, match='Frozen experiment input differs'):
        freeze_json(path, dict(seed=710001))
    assert json.loads(path.read_text()) == dict(seed=610001)


def test_test_seeds_cannot_be_used_in_development_split():
    validate_seed_split([610001], 'development')
    validate_seed_split([710001], 'test')
    with pytest.raises(ValueError):
        validate_seed_split([710001], 'development')
    with pytest.raises(ValueError):
        validate_seed_split([710001, 710001], 'test')


def test_fixed_cutoff_reports_never_departed_separately():
    metrics = normalize_metrics(dict(started_vehicle_count=8, cumulative_vehicle_seconds=1200.,
                                     vehicle_count=10, unfinished=5, never_departed=2))
    assert metrics['truncated_cost_seconds_per_started_vehicle'] == 150
    assert metrics['truncated_cost_minutes_per_started_vehicle'] == 2.5
    assert metrics['completed_vehicle_count'] == 5
    assert metrics['departed_unfinished'] == 3
