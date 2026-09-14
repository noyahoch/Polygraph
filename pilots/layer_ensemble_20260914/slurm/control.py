"""Allocated, source-bound launch and independent deadline control; no score gates."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time
from .runtime import atomic_json, file_sha256, load_workflow, require_slurm, submit_stage

ARMS = ('block2', 'block5', 'block8', 'block11')


def ledger(root):
    path = root/'manifests/submissions.tsv'
    found = {}
    for line in path.read_text().splitlines() if path.exists() else ():
        row = line.split('\t')
        if len(row) >= 3 and row[1].isdigit():
            if row[2] in found and found[row[2]] != row[1]:
                raise RuntimeError('Duplicate stage submission: ' + row[2])
            found[row[2]] = row[1]
    return found


def submit(args, stage, dependency=None):
    row = load_workflow()['stages'][stage]['resources']
    return submit_stage(args.root, args.old_root, args.release, stage,
                        row['minutes'], row['gpu'], row['cpus'], dependency)


def state_of(ids):
    if not ids:
        return {}
    result = subprocess.run(['sacct', '-j', ','.join(ids), '-P', '-n',
                             '--format=JobIDRaw,State,ExitCode,ElapsedRaw'],
                            capture_output=True, text=True, timeout=30, check=True)
    found = {}
    for line in result.stdout.splitlines():
        row = line.split('|')
        if len(row) >= 4 and row[0] in ids:
            found[row[0]] = {'state': row[1].split()[0], 'exit': row[2], 'elapsed': row[3]}
    return found


def predictions_ready(root, deadline):
    try:
        value = json.loads((root/'predictions/dev_eval.json').read_text())
        return (value.get('role') == 'dev_eval' and value.get('arms') == list(ARMS)
            and value.get('seed') == 7 and value.get('rows') == 7200
            and value.get('completed_unix', float('inf')) <= deadline.timestamp()
            and value.get('npz_sha256') == file_sha256(root/'predictions/dev_eval.npz')
            and value.get('heads_freeze_sha256') == file_sha256(root/'heads_freeze.json')
            and value.get('base_freeze_sha256') == file_sha256(root/'base_freeze.json')
            and value.get('execution_sha256') == file_sha256(root/'execution.json')
            and value.get('roles_sha256') == file_sha256(root/'role_map.json'))
    except (OSError, ValueError):
        return False


def dispatch(args):
    workflow = load_workflow()
    smoke = json.loads((args.root/'manifests/smoke.json').read_text())
    if not (smoke.get('passed') is True or smoke.get('status') in ('passed', 'complete')):
        raise RuntimeError('New role/protocol/runtime preflight did not pass')
    execution = args.root/'execution.json'
    roles = args.root/'role_map.json'
    cache = args.root/'feature_cache/manifest.json'
    if not all(p.is_file() for p in (execution, roles, cache)):
        raise RuntimeError('Execution, role and complete shared-cache identities are required')
    if (smoke.get('scope_id') != workflow['scope_id'] or set(smoke.get('arms', {})) != set(ARMS)
            or smoke.get('execution_sha256') != file_sha256(execution)
            or smoke.get('roles_sha256') != file_sha256(roles)
            or smoke.get('source_sha256') != workflow['source_sha256']['pilots/layer_ensemble_20260914/smoke.py']):
        raise RuntimeError('Preflight identity or four-arm completeness changed')
    if json.loads(execution.read_text()).get('cache_manifest_sha256') != file_sha256(cache):
        raise RuntimeError('Current full cache differs from preflight execution')
    now = dt.datetime.now(dt.timezone.utc)
    deadline = dt.datetime.fromisoformat(workflow['deadlines']['base'])
    largest = max(workflow['stages']['fit_'+arm]['resources']['minutes'] for arm in ARMS)
    admission = {'status': 'passed', 'passed': True, 'scope_id': workflow['scope_id'],
                 'execution_sha256': file_sha256(execution), 'roles_sha256': file_sha256(roles),
                 'cache_manifest_sha256': file_sha256(cache), 'smoke_sha256': file_sha256(args.root/'manifests/smoke.json'),
                 'workflow_sha256': os.environ['OMRI_WORKFLOW_SHA256'],
                 'latest_launch_for_largest_cap': (deadline-dt.timedelta(minutes=largest)).isoformat(),
                 'created_utc': now.isoformat(), 'queue_delay_unknown': True,
                 'fixed_epochs': 20, 'arms': list(ARMS), 'seed': 7}
    if now + dt.timedelta(minutes=largest) > deadline:
        admission.update(status='not_admitted', passed=False,
                         reason='Insufficient time for the predeclared largest complete-20-epoch allocation before23:00')
        atomic_json(args.root/'manifests/admission.json', admission)
        return 2
    atomic_json(args.root/'manifests/admission.json', admission)
    jobs = {arm: submit(args, 'fit_'+arm) for arm in ARMS}
    post = submit(args, 'postprocess', 'afterok:' + ':'.join(jobs.values()))
    atomic_json(args.root/'manifests/dispatch.json', {'status': 'submitted', 'scope_id': workflow['scope_id'],
                'jobs': jobs, 'postprocess_job_id': post, 'all_four_parallel': True,
                'no_score_based_admission': True, 'workflow_sha256': os.environ['OMRI_WORKFLOW_SHA256']})
    print(json.dumps({'fits': jobs, 'postprocess': post}), flush=True)
    return 0


def cancel_owned(root, stages, reason):
    jobs = ledger(root)
    ids = [jobs[stage] for stage in stages if stage in jobs]
    states = state_of(ids)
    active = [job for job, row in states.items() if row['state'] in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED')]
    if active:
        # Slurm sends TERM, waits its configured KillWait, then KILL. Trainers save
        # completed epochs atomically; an unfinished epoch never becomes complete.
        subprocess.run(['scancel', *active], check=True, timeout=30)
    receipt = {'reason': reason, 'job_ids': active, 'states_before': states,
               'utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    atomic_json(root/'manifests'/('cutoff_'+reason+'.json'), receipt)
    return receipt


def guardian(args):
    workflow = load_workflow()
    deadlines = {k: dt.datetime.fromisoformat(v) for k, v in workflow['deadlines'].items()}
    terminal_states = {'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}
    while True:
        now = dt.datetime.now(dt.timezone.utc)
        jobs = ledger(args.root)
        ids = list(jobs.values()) + [workflow['capture_job_id']]
        states = state_of(ids)
        atomic_json(args.root/'manifests/guardian.json', {'status': 'watching', 'utc': now.isoformat(),
                    'job_id': os.environ['SLURM_JOB_ID'], 'states': states, 'deadlines': workflow['deadlines']})
        if (args.root/'evaluation/complete.json').is_file():
            from .publish import finalize
            finalize(args, 'evaluation_completed')
            return 0
        base_complete = True
        for arm in ARMS:
            path = args.root/'runs'/arm/'seed7/complete.json'
            try:
                value = json.loads(path.read_text())
                base_complete &= (value.get('complete') is True and value.get('arm') == arm
                    and value.get('seed') == 7 and value.get('completed_epochs') == 20
                    and value.get('completed_unix', float('inf')) <= deadlines['base'].timestamp())
            except (OSError, ValueError):
                base_complete = False
        if now >= deadlines['base'] and not base_complete:
            cancel_owned(args.root, ['preflight', 'dispatch', *['fit_'+a for a in ARMS], 'postprocess'], 'base_deadline')
            from .publish import finalize
            finalize(args, 'base_deadline_incomplete')
            return 0
        if now >= deadlines['predictions'] and not predictions_ready(args.root, deadlines['predictions']):
            cancel_owned(args.root, ['postprocess'], 'prediction_deadline')
            from .publish import finalize
            finalize(args, 'prediction_deadline_incomplete')
            return 0
        if now >= deadlines['report_latest']:
            cancel_owned(args.root, ['postprocess'], 'report_deadline')
            from .publish import finalize
            finalize(args, 'report_deadline_incomplete')
            return 0
        required = [workflow['capture_job_id']] + [job for stage, job in jobs.items()
                    if stage in ('preflight', 'dispatch', 'postprocess') or stage.startswith('fit_')]
        failures = {job: states[job] for job in required if job in states and states[job]['state'] in terminal_states}
        if failures:
            # A sibling may still be finishing; preserve its saved state before
            # the complete-scope failure snapshot, rather than selecting survivors.
            cancel_owned(args.root, ['preflight', 'dispatch', *['fit_'+a for a in ARMS], 'postprocess'], 'required_stage_failed')
            atomic_json(args.root/'manifests/failure.json', failures)
            from .publish import finalize
            finalize(args, 'required_stage_failed')
            return 0
        post = jobs.get('postprocess')
        if post in states and states[post]['state'] == 'COMPLETED':
            from .publish import finalize
            finalize(args, 'postprocess_ended_without_complete_evaluation')
            return 0
        # A dispatcher can fail after an accepted sbatch; durable intents/ledger
        # are authoritative and ambiguous submissions are never retried here.
        time.sleep(30)


def main():
    require_slurm()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('dispatch', 'guardian'))
    for name in ('root', 'old-root', 'release'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    return dispatch(args) if args.mode == 'dispatch' else guardian(args)


if __name__ == '__main__':
    raise SystemExit(main())
