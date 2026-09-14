"""Bind the user-approved four-fit scope to prior measured caps and full cache."""
import argparse
import json
import os
from pathlib import Path

from .runtime import atomic_json, file_sha256, require_slurm

SCOPE = 'layer_screen_four_fits_20260914'
MATRIX = [('block11', 7), ('union4', 7), ('block11', 17), ('union4', 17)]
PARENT_SHA = '507b3247384ecb1cfd40654d32de163511d7d3b1af7051e9d9c2175f988ffade'


def main():
    require_slurm()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    args = p.parse_args()
    parent_path = args.root/'manifests/admission_workers_20260914.json'
    if file_sha256(parent_path) != PARENT_SHA:
        raise RuntimeError('Parent measurement report changed')
    parent = json.loads(parent_path.read_text())
    if parent.get('all18_cases_passed') is not True:
        raise RuntimeError('Original eighteen-case prerequisite failed')
    for name, sha in parent['core_files_sha256'].items():
        if file_sha256(args.release/name) != sha:
            raise RuntimeError('Numerical implementation changed: ' + name)
    cache_path = args.root/'feature_cache/manifest.json'
    cache = json.loads(cache_path.read_text())
    preview = json.loads((args.root/'preflight_cache/manifest.json').read_text())
    if cache.get('complete') is not True or cache.get('records') != 28800 or cache.get('completed_records') != 28800 or cache.get('diagnostic_only') is not False:
        raise RuntimeError('Full development cache is not complete')
    for name in ('protocol_sha256', 'cohort_sha256'):
        if cache[name] != preview[name]:
            raise RuntimeError('Capture differs from reviewed cohort/protocol: ' + name)
    if sum(shard['records'] for shard in cache['shards']) != 28800:
        raise RuntimeError('Incomplete shard index')
    fits = {}
    for arm, seed in MATRIX:
        row = dict(parent['all_runtime_scenarios'][arm + '/workers0'])
        if row['workers'] != 0 or row['cpus'] != 6 or row['limit_minutes'] != {'block11': 690, 'union4': 1080}[arm]:
            raise RuntimeError('Reviewed resource caps changed')
        fits[f'{arm}/seed{seed}'] = row
    reserved = 265 + 710 + sum(row['limit_minutes'] for row in fits.values())
    if reserved != 4515 or reserved > 4800:
        raise RuntimeError('Approved four-fit reservation exceeded')
    result = {'scope_id': SCOPE, 'status': 'complete', 'admitted': True,
        'authorization': 'User approved four-fit block11 versus union4 comparison, seeds7/17, on Slurm with80GPU-hour reservation budget.',
        'matrix': MATRIX, 'original_fourteen_fit_scope': 'deferred; not completed by this narrower comparison',
        'budget_gpu_minutes': 4800, 'budget_gpu_hours': 80, 'reserved_gpu_minutes': reserved,
        'diagnostic_cap_minutes': 265, 'extraction_limit_minutes': 710, 'max_concurrent_gpus': 8,
        'fits': fits, 'requires_full_cohort_two_fit_ramp': True,
        'initial_matrix': [('block11', 7), ('union4', 7)],
        'core_files_sha256': parent['core_files_sha256'], 'production_numerical_runtime': parent['production_numerical_runtime'],
        'source_report_sha256': PARENT_SHA, 'full_cache_manifest_sha256': file_sha256(cache_path),
        'full_cache_protocol_sha256': cache['protocol_sha256'], 'full_cache_cohort_sha256': cache['cohort_sha256'],
        'test_evaluated': False, 'job_id': os.environ['SLURM_JOB_ID']}
    atomic_json(args.root/'manifests/admission_four_20260914.json', result)
    print(json.dumps({'admitted': True, 'scope_id': SCOPE, 'reserved_gpu_minutes': reserved}), flush=True)


if __name__ == '__main__':
    main()
