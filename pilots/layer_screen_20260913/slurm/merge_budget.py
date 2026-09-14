"""Merge immutable worker evidence and estimate capped resources on Slurm CPU."""
from __future__ import annotations
import json
import math
from pathlib import Path
import statistics
from .runtime import atomic_json, file_sha256, require_slurm

ARMS = ('block2', 'block5', 'block8', 'block11', 'union4', 'union12')
WORKERS = (0, 2, 4)
MATRIX = [(arm, seed) for arm in ARMS for seed in (7, 17)] + [('block11', 1), ('block11', 27)]


def read(path):
    return json.loads(Path(path).read_text())


def complete(case):
    return (case.get('parity', {}).get('passed') is True and 'timing' in case
            and 'timing_error' not in case)


def merge(root, release):
    require_slurm()
    root, release = Path(root), Path(release)
    plan_path = release/'ops/frozen_missing_cases_892080.json'
    plan = read(plan_path)
    parent_path = root/'manifests/profile_workers_recovery_20260914.json'
    if file_sha256(parent_path) != plan['parent_result_sha256']:
        raise RuntimeError('Immutable parent result changed')
    if file_sha256(root/'preflight_cache/manifest.json') != plan['cache_manifest_sha256']:
        raise RuntimeError('Diagnostic cache changed')
    for path, wanted in plan['core_files_sha256'].items():
        if file_sha256(release/path) != wanted:
            raise RuntimeError('Unreviewed numerical helper: ' + path)
    parent = read(parent_path)
    universe = {f'{arm}/workers{worker}' for arm in ARMS for worker in WORKERS}
    done = {key for key, case in parent['cases'].items() if complete(case)}
    if done != set(plan['completed_cases']) or universe - done != set(plan['missing_cases']):
        raise RuntimeError('Frozen coverage partition changed')
    binding = {key: plan[key] for key in ('core_files_sha256', 'protocol_sha256',
        'cache_manifest_sha256', 'production_numerical_runtime', 'torch', 'gpu', 'cpus_per_task')}
    cases = {key: parent['cases'][key] for key in done}
    lineage = {key: {'job_id': plan['parent_job_id'], 'source_release': plan['parent_source_release'],
                    'result_sha256': plan['parent_result_sha256'], 'parent_terminal_state': plan['parent_terminal_state']}
               for key in done}
    sources = {str(parent_path.relative_to(root)): file_sha256(parent_path)}
    for arm in ('union12', 'block2', 'block5', 'block8', 'union4'):
        path = root/f'manifests/workers_continue_{arm}_20260914.json'
        report = read(path)
        expected = {key for key in plan['missing_cases'] if key.startswith(arm + '/')}
        if report.get('compatibility') != binding or report.get('plan_sha256') != file_sha256(plan_path):
            raise RuntimeError('Continuation lineage mismatch: ' + arm)
        if report.get('parent_result_sha256') != plan['parent_result_sha256']:
            raise RuntimeError('Continuation parent mismatch: ' + arm)
        if report.get('wrapper_sha256') != file_sha256(release/'pilots/layer_screen_20260913/continue_workers.py'):
            raise RuntimeError('Continuation wrapper changed: ' + arm)
        if set(report.get('selected_cases', [])) != expected or set(report.get('cases', {})) != expected:
            raise RuntimeError('Incomplete or duplicated continuation coverage: ' + arm)
        if report.get('selected_cases_passed') is not True or report.get('error'):
            raise RuntimeError('Continuation is not complete: ' + arm)
        sources[str(path.relative_to(root))] = file_sha256(path)
        for key, payload in report['cases'].items():
            if key in cases or payload.get('key') != key or payload.get('compatibility') != binding:
                raise RuntimeError('Duplicate/incompatible case: ' + key)
            if payload.get('plan_sha256') != file_sha256(plan_path) or payload.get('parent_result_sha256') != plan['parent_result_sha256']:
                raise RuntimeError('Case lineage changed: ' + key)
            case = payload['case']
            if not complete(case):
                raise RuntimeError('Failed/incomplete case: ' + key)
            cases[key] = case
            lineage[key] = {k: payload[k] for k in ('job_id', 'wrapper_sha256', 'plan_sha256', 'parent_result_sha256')}
    if set(cases) != universe:
        raise RuntimeError('Exactly eighteen complete cases are required')
    for key, case in cases.items():
        if case['timing']['numerical_runtime'] != binding['production_numerical_runtime']:
            raise RuntimeError('Timing backend mismatch: ' + key)
        if case['restored_numerical_runtime'] != binding['production_numerical_runtime']:
            raise RuntimeError('Audit did not restore production backend: ' + key)
        if case['parity_numerical_runtime']['deterministic_algorithms'] is not True:
            raise RuntimeError('Parity audit used the wrong backend: ' + key)
    result = {'complete': True, 'production_admitted': False, 'cases': cases,
              'case_provenance': lineage, 'compatibility': binding,
              'parent_terminal_state': plan['parent_terminal_state'],
              'parent_overall_passed': parent.get('passed'),
              'source_files_sha256': sources, 'plan_sha256': file_sha256(plan_path),
              'limitations': parent.get('limitations', [])}
    atomic_json(root/'manifests/worker_cases_merged_20260914.json', result)
    return result


def estimate(root, merged):
    """Conservative sample-based scenario; full-cohort ramp is still required."""
    require_slurm()
    root = Path(root)
    profile = read(root/'manifests/profile_timing_20260914.json')
    preflight = read(root/'preflight.json')
    capture = read(root/'preflight_cache/capture_summary.json')
    execution = read(root/'manifests/execution/profile_timing_20260914_891747.json')
    if not profile.get('passed') or not preflight.get('passed'):
        raise RuntimeError('Original allocated timing/integrity evidence missing')
    cases = merged['cases']
    hash_times = [e['seconds'] for e in profile['io_events'] if e['kind'] == 'verification'
                  and e['first_observed_in_process'] and str(e['path']).endswith('.pt')]
    if not hash_times:
        raise RuntimeError('No measured first-file checksum cost')
    shard_ratio = 64 / 36
    checksum64 = max(hash_times) * shard_ratio
    bootstrap = max(0, execution['elapsed_seconds'] - profile['elapsed_seconds'])
    cuda_excess = max(0, max(v['train_batches'][0]['train_step_seconds'] -
                            v['warm_summary']['train_step_seconds']['median']
                            for v in profile['arms'].values()))
    parallel_val_lifecycle = max(cases[f'{a}/workers{w}']['timing']['validation_lifecycle_seconds']
                                 for a in ARMS for w in (2, 4))
    rows = {}
    for arm in ARMS:
        old = profile['arms'][arm]
        # This sums new-shard service time without assuming unmeasured overlap.
        deserialize64 = max(row['deserialization_seconds'] for row in old['reloads']) * shard_ratio
        memo_times = [e['seconds'] for e in profile['io_events'] if e['kind'] == 'verification'
                      and not e['first_observed_in_process'] and str(e['phase']).startswith(arm + '/')]
        memo64 = max(memo_times, default=0)
        serialized = old['serialization']
        checkpoint = serialized['snapshot_construction_seconds'] + sum(
            serialized[name]['write_seconds'] + serialized[name]['hash_seconds'] for name in ('latest', 'best'))
        export = serialized['validation_export']['write_seconds'] + serialized['validation_export']['hash_seconds']
        serial = cases[f'{arm}/workers0']['timing']
        audit_batch = max(row['end_to_end_seconds'] * 24 / row['records'] for row in serial['validation_batches'])
        for workers in WORKERS:
            observed = cases[f'{arm}/workers{workers}']['timing']
            warm = observed['warm_24_consecutive_wall_seconds_per_batch']
            if not math.isfinite(warm) or warm <= 0:
                raise RuntimeError('Invalid warm wall time')
            processes = max(1, workers)
            val_startup = parallel_val_lifecycle if workers else observed['validation_lifecycle_seconds']
            reloads = processes * (338 + 113) + 2
            val_hash_files = workers * 113
            initial_hash_files = workers * 338 + 2 if workers else 450
            new_io = reloads * (deserialize64 + 2 * memo64)
            recurring_hash = val_hash_files * checksum64
            cycle = 900 * warm + 300 * warm + val_startup + 2 * audit_batch + new_io + recurring_hash + checkpoint
            # Constructor already reads the complete cohort even for the tiny cache.
            # Charge two observed constructions once, plus the prior setup reserve.
            startup = (bootstrap + 2 * old['dataset_construction_seconds'] +
                       old['model_optimizer_setup_seconds'] + cuda_excess +
                       initial_hash_files * checksum64 + observed['train_batches'][0]['end_to_end_seconds'] + 1800)
            final = ((300 * warm + val_startup + workers * 113 * checksum64 +
                          processes * 113 * deserialize64) + 6 * audit_batch + 6 * deserialize64 + export)
            scenario = startup + 60 * cycle + final
            minutes = math.ceil(1.5 * scenario / 300) * 5
            optimistic = scenario - initial_hash_files * checksum64 - 60 * (new_io + recurring_hash) - (workers * 113 * checksum64 + processes * 113 * deserialize64) - 6 * deserialize64
            optimistic_minutes = math.ceil(1.5 * optimistic / 300) * 5
            memory_ok = (observed['sampled_process_rss_sum_peak_bytes'] * 1.5 <= 32000 * 1024**2 and
                         observed['gpu_max_allocated_bytes'] * 1.5 <= preflight['total_gpu_bytes'])
            rows[f'{arm}/workers{workers}'] = {
                'workers': workers, 'cpus': 6, 'limit_minutes': minutes,
                'startup_final_reserve_seconds': 1800,
                'warm_train_seconds_per24': warm,
                'warm_eval_proxy_seconds_per24': warm,
                'validation_lifecycle_upper_seconds': val_startup,
                'serial_audit_seconds_per_epoch': 2 * audit_batch,
                'reload_count_per_epoch': reloads,
                'deserialize64_seconds': deserialize64,
                'new_shard_io_seconds_per_epoch': new_io,
                'validation_first_hash_seconds_per_epoch': recurring_hash,
                'checkpoint_seconds_per_epoch': checkpoint,
                'once_per_fit_startup_sensitivity_seconds': startup,
                'finalization_sensitivity_seconds': final,
                'whole_epoch_scenario_seconds': cycle,
                'raw_scenario_seconds': scenario,
                'optimistic_no_new_io_cap_minutes': optimistic_minutes,
                'sample_memory_headroom_passed': memory_ok,
                'sampled_rss_sum_bytes': observed['sampled_process_rss_sum_peak_bytes'],
                'sample_gpu_max_allocated_bytes': observed['gpu_max_allocated_bytes']}
    # Compare runtime settings only; no model outcomes enter this selection.
    settings = {}
    for worker in WORKERS:
        fit_minutes = sum(rows[f'{arm}/workers{worker}']['limit_minutes'] for arm, seed in MATRIX)
        settings[str(worker)] = {'fit_cap_minutes': fit_minutes,
                                'optimistic_no_new_io_fit_cap_minutes': sum(rows[f'{arm}/workers{worker}']['optimistic_no_new_io_cap_minutes'] for arm, seed in MATRIX),
                                'memory_passed': all(rows[f'{arm}/workers{worker}']['sample_memory_headroom_passed'] for arm in ARMS)}
    eligible = [w for w in WORKERS if settings[str(w)]['memory_passed']]
    if not eligible:
        raise RuntimeError('No measured worker setting fits allocated memory with headroom')
    chosen = min(eligible, key=lambda w: settings[str(w)]['fit_cap_minutes'])
    # Retain the original gross extraction cap; no warm capture timing exists.
    capture_minutes = math.ceil((capture['startup_seconds'] + 1.5 * capture['projected_28800_seconds'] + 1800) / 300) * 5
    total = 265 + capture_minutes + settings[str(chosen)]['fit_cap_minutes']
    storage_ok = capture['projected_28800_bytes'] * 2 <= capture['free_bytes']
    fits = {f'{arm}/seed{seed}': dict(rows[f'{arm}/workers{chosen}']) for arm, seed in MATRIX}
    result = {'admitted': total <= 3840 and storage_ok, 'requires_full_cohort_two_fit_ramp': True,
        'all18_cases_passed': True, 'budget_gpu_hours': 64,
        'diagnostic_cap_minutes': 265, 'extraction_limit_minutes': capture_minutes,
        'projected_reserved_gpu_hours': total / 60, 'chosen_workers': chosen,
        'optimistic_no_new_io_total_gpu_hours_by_workers': {str(w): (265 + capture_minutes + settings[str(w)]['optimistic_no_new_io_fit_cap_minutes']) / 60 for w in WORKERS},
        'fits': fits, 'settings': settings, 'all_runtime_scenarios': rows,
        'matrix': MATRIX, 'max_concurrent_gpus': 8,
        'core_files_sha256': merged['compatibility']['core_files_sha256'],
        'production_numerical_runtime': merged['compatibility']['production_numerical_runtime'],
        'sample_storage_headroom_passed': storage_ok,
        'corrected_source_accounting': {'cohort_constructor_multiplier': 1, 'full_final_validation_passes': 1},
        'reason': 'Within nominal GPU caps; mandatory full-cohort ramp remains' if total <= 3840 and storage_ok
                  else 'Conservative complete-matrix scenario exceeds budget or storage bound',
        'limitations': [
            'One36-record shard does not establish full-cache or8-job filesystem throughput.',
            'New-shard I/O and recurring validation hashes are charged without assumed overlap; this is a conservative scenario, not a measured full-fit runtime.',
            'Warm training wall proxies validation computation; validation36-row lifecycle is added once per epoch, intentionally double-counting its tiny evaluation component.',
            'Maximum observed parallel validation lifecycle is preserved across worker settings; startup variation is not discarded.',
            'Extraction cap is the legacy gross36-record extrapolation with unknown first-use CUDA cost, not a warmed rate.',
            'Dataset constructor already reads the full cohort; measured construction is charged once per train/validation dataset, with the prior30-minute setup reserve.',
            'All60epochs and submitted failed diagnostic caps are charged; early stopping is not assumed.',
            'Memory is sampled RSS sum and allocated CUDA memory, not a guarantee of full-cohort peaks; scheduler limits and the two-fit ramp remain.']}
    result['inputs_sha256'] = {str(path.relative_to(root)): file_sha256(path) for path in (
        root/'manifests/profile_timing_20260914.json', root/'preflight.json',
        root/'preflight_cache/capture_summary.json', root/'manifests/worker_cases_merged_20260914.json')}
    atomic_json(root/'manifests/admission_workers_20260914.json', result)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    args = parser.parse_args()
    require_slurm()
    try:
        merged = merge(args.root, args.release)
        result = estimate(args.root, merged)
        print(json.dumps({key: result[key] for key in ('admitted', 'projected_reserved_gpu_hours',
            'chosen_workers', 'settings', 'optimistic_no_new_io_total_gpu_hours_by_workers', 'diagnostic_cap_minutes', 'extraction_limit_minutes', 'reason')}), flush=True)
    except BaseException as error:
        atomic_json(args.root/'manifests/admission_workers_20260914.json', {
            'admitted': False, 'status': 'evidence_or_estimator_failed', 'error': repr(error),
            'diagnostic_cap_minutes': 265, 'production_jobs_submitted': 0})
        raise


if __name__ == '__main__':
    main()
