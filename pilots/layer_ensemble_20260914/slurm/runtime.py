"""Small stdlib-only helpers; queue mutations require an allocated reviewed job."""
from __future__ import annotations
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('w') as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_slurm():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('An allocated Slurm job is required')


def load_workflow():
    require_slurm()
    path = Path(os.environ['OMRI_WORKFLOW_PATH'])
    if file_sha256(path) != os.environ['OMRI_WORKFLOW_SHA256']:
        raise RuntimeError('Workflow changed after submission')
    return json.loads(path.read_text())


def submit_stage(root, old, release, stage, minutes, gpu, cpus, dependency=None):
    """Submit one declared stage, preserve its actual ID, and never shell-expand it."""
    require_slurm()
    root, old, release = Path(root), Path(old), Path(release)
    workflow = load_workflow()
    if release.resolve() != Path(workflow['code_root']).resolve():
        raise RuntimeError('Undeclared source release')
    if stage not in workflow['stages'] or int(minutes) != minutes or not 1 <= minutes <= 4320:
        raise RuntimeError('Undeclared stage or invalid time cap')
    if not 1 <= int(cpus) <= 6:
        raise RuntimeError('Unexpected CPU request')
    args = ['sbatch', '--parsable', '--account=gpu-students',
            '--partition=' + ('studentbatch' if gpu else 'cpu-killable'),
            '--cpus-per-task=' + str(cpus), '--mem=' + ('32000M' if gpu else '8000M'),
            '--time=' + str(minutes), '--job-name=omri-layers-' + stage,
            '--no-requeue', '--kill-on-invalid-dep=yes',
            '--output=' + str(root/'logs'/f'{stage}_%j.out'),
            '--error=' + str(root/'logs'/f'{stage}_%j.err')]
    if gpu:
        args += ['--gpus=1', '--constraint=geforce_rtx_2080']
    if dependency:
        args += ['--dependency=' + str(dependency)]
    args += [str(release/'pilots/layer_ensemble_20260914/slurm/stage.sbatch'),
             str(root), str(old), stage, str(release)]
    declared = workflow['stages'][stage]['resources']
    if (int(minutes), bool(gpu), int(cpus)) != (declared['minutes'], declared['gpu'], declared['cpus']):
        raise RuntimeError('Stage resources differ from the frozen workflow')
    if workflow['historical_diagnostic_cap_minutes'] + workflow['capture_cap_minutes'] + sum(
            row['resources']['minutes'] for row in workflow['stages'].values() if row['resources']['gpu']) > workflow['budget_gpu_minutes']:
        raise RuntimeError('Frozen allocation reservations exceed the user budget')
    intents = root/'manifests/submission_intents'
    intents.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(stage.encode()).hexdigest()
    intent_path = intents/(key + '.json')
    with (intents/(key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if intent_path.exists():
            previous = json.loads(intent_path.read_text())
            if previous.get('status') == 'recorded' and str(previous.get('job_id', '')).isdigit():
                if previous.get('command') != args:
                    raise RuntimeError('Stage already submitted with different arguments: ' + stage)
                return str(previous['job_id'])
            raise RuntimeError('Unresolved submission intent; reconcile with Slurm before resubmission: ' + str(intent_path))
        intent = {'status': 'submitting', 'stage': stage, 'command': args,
                  'workflow_sha256': os.environ['OMRI_WORKFLOW_SHA256'],
                  'created_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
        atomic_json(intent_path, intent)
        # Any interruption after this point remains ambiguous until reconciled.
        job = subprocess.check_output(args, text=True).strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Unrecognized sbatch receipt: ' + repr(job))
        with (root/'manifests/submissions.tsv').open('a') as output:
            fcntl.flock(output, fcntl.LOCK_EX)
            output.write(f'{dt.datetime.now(dt.timezone.utc).isoformat()}\t{job}\t{stage}\t{minutes}min\t{int(gpu)}gpu\t{cpus}cpu\n')
            output.flush()
            os.fsync(output.fileno())
        intent.update(status='recorded', job_id=job)
        atomic_json(intent_path, intent)
    return job
