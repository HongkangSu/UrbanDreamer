"""Verify one full-network event FM forward/backward from each frozen city manifest."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import resource
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from urban_dreamer.learning.action_events import pack_action_events
from urban_dreamer.learning.offline_data import OfflineEpisodeDataset, archive_content_hash
from urban_dreamer.models.world_model import UrbanDreamerWorldModel, recursive_fm_loss
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config


CITIES = ('gretz_armainvilliers', 'fontenay_en_parisis', 'ingolstadt_custom')


def verify_city(city: str, manifest: Path, verification_manifest: Path) -> dict:
    started = time.monotonic()
    torch.manual_seed(42)
    config = load_config(ROOT / 'configs' / f'{city}.toml')
    config = replace(config, run=replace(config.run, device='cpu'))
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    frozen = json.loads(manifest.read_text())
    for entry in frozen['files']:
        if archive_content_hash(Path(entry['path'])) != entry['content_hash']:
            raise ValueError(f"frozen source archive content changed: {entry['path']}")
    dataset = OfflineEpisodeDataset(config, prepared, topology,
        sequence_length=1, archive_paths=[entry['path'] for entry in frozen['entries']],
        manifest_path=verification_manifest)
    if not dataset.has_samples:
        raise ValueError(f'{city}: manifest has no archived transitions')
    # Select one already frozen source and its most eventful macro window, without sampling new data.
    episode = dataset._episode(0)
    rows = episode['transitions']
    index = max(range(len(rows)), key=lambda i: len(rows[i].action_events))
    row = rows[index]
    packed = pack_action_events([[row]])
    if not len(packed):
        raise AssertionError(f'{city}: the sampled episode must exercise event conditioning')
    model = UrbanDreamerWorldModel(adjacency=torch.as_tensor(topology.network.adjacency),
        static_features=torch.as_tensor(topology.network.static_features), event_protocol=True)
    # Existing zero velocity heads intentionally block condition gradients on the initial update.
    # A small deterministic nonzero head verifies the event parameter training path directly.
    torch.nn.init.normal_(model.dynamics.flow_head.weight, std=0.01)
    torch.nn.init.normal_(model.dynamics.graph_prior.output.weight, std=0.01)
    flow = torch.stack([row.flow, row.next_flow])[None]
    latent = model.encode(flow)
    loss, metrics, predicted = recursive_fm_loss(model.dynamics, z_sequence=latent,
        action=row.action[None, None], action_events=packed, steps=1,
        flow_time=torch.full((1, 1), 0.5))
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(predicted).all()):
        raise AssertionError(f'{city}: nonfinite FM forward pass')
    loss.backward()
    gradients = {}
    encoder = model.dynamics.event_encoder
    for name, parameter in (('edge_embedding', encoder.edge_embedding.weight),
                            ('event_mlp_input', encoder.event_mlp[0].weight),
                            ('event_mlp_output', encoder.event_mlp[2].weight)):
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()) or gradient.abs().sum() <= 0:
            raise AssertionError(f'{city}: {name} requires finite nonzero FM gradients')
        gradients[name] = dict(l1=float(gradient.abs().sum()), l2=float(gradient.norm()),
                               max_abs=float(gradient.abs().max()))
    result = dict(status='passed', city=city, device='cpu', batch_size=1, supervision_depth=1,
        gradient_verification_head_std=0.01,
        edge_count=topology.network.edge_count, window_index=index, window_start=row.time,
        sampled_events=len(packed), sampled_population=episode['total_vehicle_count'],
        archive=dataset.manifest['entries'][0]['path'], manifest_identity=dataset.manifest['identity'],
        source_manifest_identity=frozen['identity'], target_deadline_seconds=dataset.target_deadline_seconds,
        import_summary=dataset.summary, loss=float(loss.detach()),
        flow_mse=float(metrics['flow_mse']), prior_mse=float(metrics['prior_mse']),
        event_gradients=gradients, elapsed_seconds=time.monotonic() - started,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        note='Full city graph and real archived events; functional finite-value/gradient check, not traffic performance evidence.')
    del model, latent, loss, predicted, dataset, topology, rows, row, flow
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--city', choices=CITIES, action='append')
    parser.add_argument('--manifest-dir', type=Path, default=Path('/tmp'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for city in args.city or CITIES:
        print(json.dumps(dict(city=city, status='running')), flush=True)
        result = verify_city(city, args.manifest_dir / f'urban_offline_verify_aliases_{city}.json',
                             args.output_dir / f'{city}_offline_manifest.json')
        (args.output_dir / f'{city}.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
