"""Stop one identity-verified online process group after its next committed episode."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time


def process(pid):
    path = Path('/proc') / str(pid)
    try:
        stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        command = (path / 'cmdline').read_bytes().split(b'\0')
        return dict(identity=stat[19], state=stat[0],
                    command=[x.decode() for x in command if x])
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--record-dir', type=Path, required=True)
    parser.add_argument('--replacement-output', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=float, default=600.)
    parser.add_argument('--after-completed', type=int,
                        help='resume a previous stop request after this already-observed episode count')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    status_path = root / 'online_supervisor/status.json'
    status = json.loads(status_path.read_text())
    output = Path(status['binding']['output_dir'])
    pid, watcher = int(status['child_pid']), int(status['watcher_pid'])
    child = process(pid)
    if (not child or child['identity'] != status['child_identity']
            or str(output) not in child['command'] or os.getpgid(pid) != pid):
        raise ValueError('online child identity/output/process group mismatch')
    supervisor = process(watcher)
    if supervisor and (not any(x.endswith('/queue_pretrained_online.py') for x in supervisor['command'])
                       or str(output) not in supervisor['command']):
        raise ValueError('supervisor identity mismatch')
    bridge_status = json.loads((root/'status.json').read_text())
    bridge_pid = bridge_status.get('controller_pid')
    bridge = process(bridge_pid) if bridge_pid else None
    if bridge and (not any(x.endswith('/continue_mainline_paper_reward.py') for x in bridge['command'])
                   or str(root/'paper_reward_future_plan.json') not in bridge['command']):
        raise ValueError('paper handoff bridge identity mismatch')
    progress_path = output / 'pilot_progress.json'
    observed = len(json.loads(progress_path.read_text())['online'])
    initial = observed if args.after_completed is None else args.after_completed
    if initial < 0 or initial > observed:
        raise ValueError('resume boundary must have already been observed')
    record = dict(child_pid=pid, child_identity=child['identity'], watcher_pid=watcher, bridge_pid=bridge_pid,
                  source_output=str(output), initial_completed=initial,
                  replacement_output=str(args.replacement_output.resolve()),
                  requested_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
    print(json.dumps(record), flush=True)
    if args.dry_run:
        return
    args.record_dir.mkdir(parents=True, exist_ok=False)
    (args.record_dir/'request.json').write_text(json.dumps(record, indent=2)+'\n')
    (args.record_dir/'original_supervisor_status.json').write_text(json.dumps(status, indent=2)+'\n')
    if bridge:
        # Bridge's handler terminates only its queue child and records a clean
        # bridge stop. Queue's handler explicitly leaves online training alive.
        os.kill(bridge_pid, signal.SIGTERM)
    elif supervisor:
        os.kill(watcher, signal.SIGTERM)  # Its handler leaves the child running.
    for _ in range(600):
        supervisor_now = process(watcher)
        if supervisor_now is None or supervisor_now['state'] == 'Z':
            break
        time.sleep(.1)
    else:
        raise RuntimeError('supervisor did not stop; online child was not signalled')
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        current = process(pid)
        if current is None or current['identity'] != child['identity']:
            raise RuntimeError('online child exited before the requested checkpoint boundary')
        progress = json.loads(progress_path.read_text())
        if len(progress['online']) > initial:
            break
        time.sleep(.2)
    else:
        raise TimeoutError('no committed episode observed; child was not signalled')
    # Save has atomically replaced latest.pt before progress is published.
    # A tiny beginning of the next episode may already be in flight; only the
    # committed checkpoint is transferred, never unfinished optimizer state.
    os.killpg(pid, signal.SIGSTOP)
    time.sleep(.2)
    current = process(pid)
    if not current or current['identity'] != child['identity'] or current['state'] not in ('T', 't'):
        raise RuntimeError('could not verify frozen online process')
    frozen_progress = json.loads(progress_path.read_text())
    shutil.copyfile(output/'checkpoints/latest.pt', args.record_dir/'checkpoint.pt')
    shutil.copyfile(output/'run_config.toml', args.record_dir/'run_config.toml')
    (args.record_dir/'pilot_progress.json').write_text(json.dumps(frozen_progress, indent=2)+'\n')
    os.killpg(pid, signal.SIGTERM)
    os.killpg(pid, signal.SIGCONT)
    for _ in range(100):
        current = process(pid)
        if current is None or current['state'] == 'Z':
            break
        time.sleep(.1)
    else:
        os.killpg(pid, signal.SIGKILL)
        record['forced_group_kill'] = True
    record.update(completed_episodes=len(frozen_progress['online']), phase='stopped_and_snapshotted',
                  stopped_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
    (args.record_dir/'stop_record.json').write_text(json.dumps(record, indent=2)+'\n')
    status = json.loads(status_path.read_text())
    status.update(phase='stopped_for_bounded_dynamic_migration', child_left_running=False,
                  replacement_output=record['replacement_output'])
    status_path.write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
