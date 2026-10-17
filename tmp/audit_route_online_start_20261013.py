"""Audit actual episode updates, workers and real periodic route decisions."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from urban_dreamer.workflow.config import load_config

BATCH = ROOT / 'outputs/urbandreamer_current_to_dest_1000_fp16_w8_r8_p2_20261013'
torch.set_num_threads(1)
rows = []
for directory in (BATCH / city for city in ('ingolstadt_custom', 'coulommiers', 'montereau_fault_yonne')):
    status = json.loads((directory / 'status.json').read_text())
    row = dict(city=directory.name, pid=status['pid'], status=status['status'],
               phase=status.get('phase'), completed_episodes=status.get('completed_episodes', 0))
    assert row['status'] != 'failed', directory / 'online.log'
    output = directory / 'online'
    config_path = output / 'run_config.toml'
    if config_path.exists():
        config = load_config(config_path)
        assert config.simulation.routing_mode == 'current_to_dest'
        assert config.run.online_training_episodes == 1000
        assert config.resources.imagination_passes_per_episode == 2
        assert config.resources.imagination_rollouts_per_pass == 8
        assert config.resources.world_model_imagination_precision == 'fp16'
        assert config.resources.imagination_time_budget_seconds == 0
        protocol = json.loads((output / 'pilot_protocol.json').read_text())
        assert protocol['routing_initialization']['world_model_tensors_unchanged']
        assert protocol['online_episode_budget'] == 1000
        row['configuration_verified'] = True
    progress_path = output / 'pilot_progress.json'
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        row.update(phase=progress['phase'], completed_episodes=len(progress['online']),
                   completed_before_evaluations=len(progress['before']))
        if progress['online']:
            first = progress['online'][0]
            assert first['metrics']['imagination_passes_completed'] == 2
            assert first['optimizer_steps'] == dict(actor=2, event_critic=2, latent_critic=2)
            row.update(first_episode_seconds=first['seconds'],
                       first_optimizer_steps=first['optimizer_steps'])
            raw = torch.load(output / 'checkpoints/latest.pt', map_location='cpu', weights_only=True)
            metrics = raw['training_state']['metrics']
            assert metrics['imagination_workers'] == 8
            assert metrics['world_model_imagination_execution_profile']['precision'] == 'fp16'
            row.update(checkpoint_next_unit=raw['next_unit'], imagination_workers=8,
                       imagination_execution_profile=metrics['world_model_imagination_execution_profile'])
            del raw
            workers = []
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                    command = (proc / 'cmdline').read_bytes()
                    if fields[0] != 'Z' and int(fields[1]) == status['pid'] and b'spawn_main' in command:
                        workers.append(int(proc.name))
                except FileNotFoundError:
                    pass
            assert len(workers) == 8, workers
            row['worker_pids'] = workers
            archive = output / 'episodes/online/episode_000000.npz'
            with np.load(archive, allow_pickle=False) as data:
                metadata = json.loads(data['summary_json'].tobytes())['metadata']
                choices = json.loads(data['policy_decisions_json'].tobytes())
                assert metadata['routing_mode'] == 'current_to_dest'
                assert choices and all(c['decision_time'] % 30 == 0 for c in choices)
                assert all(c['route'][0] == c['current_edge'] and
                           c['route'][-1] == c['destination_edge'] for c in choices)
                row.update(real_policy_decisions=len(choices),
                           all_real_policy_decisions_on_30s_boundary=True,
                           all_routes_current_edge_to_destination=True)
            row['first_episode_verified'] = True
    rows.append(row)
report = dict(verified_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
              all_first_episodes_verified=all(row.get('first_episode_verified') for row in rows), cities=rows)
(BATCH / 'startup_verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
