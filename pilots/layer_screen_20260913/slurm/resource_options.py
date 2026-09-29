"""Read-only resource alternatives; no scheduler mutations or scientific changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm allocation required')
    src = a.root/'manifests/admission_workers_20260914.json'
    report = json.loads(src.read_text())
    rows = report['all_runtime_scenarios']
    arms = ['block2', 'block5', 'block8', 'block11', 'union4', 'union12']
    best = {arm: min((rows[f'{arm}/workers{w}'] for w in (0, 2, 4) if rows[f'{arm}/workers{w}']['sample_memory_headroom_passed']), key=lambda r: r['limit_minutes']) for arm in arms}
    overhead = report['diagnostic_cap_minutes'] + report['extraction_limit_minutes']
    choices = {
        'fixed_initial_two_fits': [('block11', 7), ('union12', 7)],
        'block11_union12_two_seeds_each': [(arm, seed) for arm in ('block11', 'union12') for seed in (7, 17)],
        'four_single_layers_two_seeds_each': [(arm, seed) for arm in arms[:4] for seed in (7, 17)],
        'four_single_layers_one_seed_each': [(arm, 7) for arm in arms[:4]],
        'complete_original14': report['matrix'],
    }
    output = {'admission_input_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
              'job_id': os.environ['SLURM_JOB_ID'], 'scope': 'options only; no authorization or submission',
              'diagnostic_cap_minutes': report['diagnostic_cap_minutes'],
              'capture_cap_minutes': report['extraction_limit_minutes'],
              'capture_gross_extrapolated_hours_without_headroom': json.loads((a.root/'preflight_cache/capture_summary.json').read_text())['projected_28800_seconds']/3600,
              'arms': {arm: {'best_conservative_workers': best[arm]['workers'],
                            'per_fit_cap_minutes': best[arm]['limit_minutes'],
                            'raw_conservative_scenario_hours_before_headroom': best[arm]['raw_scenario_seconds']/3600,
                            'all_worker_cap_minutes': {str(w): rows[f'{arm}/workers{w}']['limit_minutes'] for w in (0, 2, 4)},
                            'workers4_optimistic_no_new_io_cap_minutes': rows[f'{arm}/workers4']['optimistic_no_new_io_cap_minutes']} for arm in arms},
              'options': {}}
    for name, matrix in choices.items():
        caps = sum(best[arm]['limit_minutes'] for arm, seed in matrix)
        raw = sum(best[arm]['raw_scenario_seconds'] for arm, seed in matrix)
        optimistic4 = sum(rows[f'{arm}/workers4']['optimistic_no_new_io_cap_minutes'] for arm, seed in matrix)
        output['options'][name] = {'fits': matrix, 'fit_cap_hours': caps/60,
            'total_cap_hours_including_overhead': (overhead+caps)/60,
            'raw_fit_scenario_hours_before_headroom': raw/3600,
            'within64_cap': overhead+caps <= 3840,
            'workers4_optimistic_total_cap_hours': (overhead+optimistic4)/60}
    output['caveats'] = ['Caps include 1.5 safety multiplier and five-minute upward rounding; these are not measured actual full-fit runtimes.',
        'The conservative scenario serially charges new-shard I/O and validation hashes; its full-cache throughput is unmeasured.',
        'Optimistic scenario removes new-shard I/O/checksums, but retains setup reserves and safety headroom; it is not a lower bound on actual runtime.',
        'Changing fit matrix is a scientific scope decision; this script cannot authorize it.',
        'A first-two-fit resource measurement alone cannot establish comparative scientific results or authorize the remaining twelve fits.']
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
