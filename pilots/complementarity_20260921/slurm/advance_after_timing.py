"""Durable, metadata-only throughput decision after fifty retained fixed draws."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', default=os.environ.get('COMPLEMENTARITY_CONFIG_SHA256'))
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Timing continuation is allocation-only')
    actual = hashlib.sha256(args.config.read_bytes()).hexdigest()
    if actual != args.config_sha256:
        raise RuntimeError('Configuration changed before continuation')
    config = json.loads(args.config.read_text())
    root = Path(config['root'])
    timing_path = root / 'statistics/timing_50.json'
    timing = json.loads(timing_path.read_text())
    if timing.get('campaign_sha256') != hashlib.sha256((root / 'campaign.json').read_bytes()).hexdigest():
        raise RuntimeError('Timing evidence is bound to a different frozen campaign')
    # No prediction, metric, bootstrap-effect or model files are read here.
    count = int(timing['retained_draw_count'])
    ids = timing['completed_draw_ids']
    elapsed = float(timing['elapsed_seconds'])
    if timing.get('complete') is not True or count != 50 or ids != list(range(50)) or not elapsed > 0:
        raise RuntimeError('Timing pilot must contain exactly the first fifty retained draws')
    remaining = elapsed * 1950 / 50
    safety_estimate = 1.25 * remaining
    budget = config['stages']['statistics_rest']['minutes'] * 60
    unexpected = int(timing['unexpected_error_draw_count'])
    continuation_allowed = safety_estimate <= budget and unexpected == 0
    gate = dict(complete=True, within_budget=safety_estimate <= budget,
                continuation_allowed=continuation_allowed,
                unexpected_error_draw_count=unexpected,
                failure_draw_count=int(timing['failure_draw_count']),
                checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                configuration_sha256=actual,
                timing_sha256=hashlib.sha256(timing_path.read_bytes()).hexdigest(),
                retained_draw_ids=ids, measured_seconds=elapsed,
                predicted_remaining_seconds=remaining,
                safety_factor=1.25, conservative_remaining_seconds=safety_estimate,
                fixed_remaining_allocation_seconds=budget,
                action='continue' if continuation_allowed else
                       ('stop_for_technical_review' if unexpected else 'stop_for_budget_review'),
                scientific_effects_inspected=False)
    gate_path = root / 'ops/timing_gate.json'
    with gate_path.open('x') as stream:
        json.dump(gate, stream, indent=2, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(gate), flush=True)
    if not gate['continuation_allowed']:
        return
    subprocess.run([config['python'], '-B', str(Path(config['release']) / 'ops/complementarity_submit.py'),
                    '--config', str(args.config), '--approved-config-sha256', actual,
                    '--phase', 'remaining'], check=True)


if __name__ == '__main__':
    main()
