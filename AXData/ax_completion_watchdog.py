"""External worker-pool watchdog; does not change model/data/campaign hashes."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def broken_pool(state):
    rows = state.get('Completed_Lanes', []) + state.get('Records', [])
    return any('BrokenProcessPool' in str(row.get('error', '')) for row in rows)


def run(command, state_dir, interval=5, grace=15):
    started = time.time()
    child = subprocess.Popen(command, start_new_session=True)
    interrupted = []
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, lambda number, frame: interrupted.append(number))
    reason = None
    try:
        while child.poll() is None:
            if interrupted:
                reason = 'watchdog_stop_requested'
                break
            for name in ('scheduler.json', 'test_scheduler.json'):
                path = state_dir / name
                try:
                    if path.stat().st_mtime < started:
                        continue
                    state = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                if broken_pool(state):
                    reason = 'broken_process_pool'
                    break
            if reason:
                break
            time.sleep(interval)
        if reason:
            print(f'[watchdog] {reason}; stop owned process group {child.pid}', flush=True)
            os.killpg(child.pid, signal.SIGTERM)
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                try:
                    os.killpg(child.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(min(interval, 1))
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            record = {'reason': reason, 'pid': child.pid, 'at': time.time(), 'returncode': child.returncode}
            with (state_dir / 'watchdog_events.jsonl').open('a') as out:
                out.write(json.dumps(record) + '\n')
            return 75 if not interrupted else 128 + interrupted[0]
        return child.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--grace', type=float, default=15)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not cmd:
        parser.error('command required')
    raise SystemExit(run(cmd, args.state_dir, args.interval, args.grace))
