"""Persist a complete or explicitly partial report and private reproducibility copy."""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from .runtime import atomic_json, file_sha256, load_workflow


def finalize(args, reason):
    root = args.root
    publication = root/'publication'
    publication.mkdir(parents=True, exist_ok=True)
    report = root/'evaluation/report.json'
    status = 'partial'
    evaluation_complete = root/'evaluation/complete.json'
    if report.is_file() and evaluation_complete.is_file():
        result = json.loads(report.read_text())
        completed = json.loads(evaluation_complete.read_text())
        expected_files = {'report.json', 'REPORT.md', 'bootstrap.json', 'bootstrap_source_counts.npz', 'scores.npz'}
        verified = (completed.get('complete') is True
            and set(completed.get('files', {})) == expected_files
            and all((root/'evaluation'/name).is_file()
                    and file_sha256(root/'evaluation'/name) == completed['files'][name] for name in expected_files))
        from .control import predictions_ready
        deadlines = load_workflow()['deadlines']
        ready_on_time = predictions_ready(root, dt.datetime.fromisoformat(deadlines['predictions']))
        report_on_time = dt.datetime.fromisoformat(result.get('completed_utc', '9999-01-01T00:00:00+00:00')) <= dt.datetime.fromisoformat(deadlines['report_latest'])
        if verified and ready_on_time and report_on_time and result.get('complete') is True and result.get('status') == 'complete':
            status = 'complete'
    state = {'status': status, 'reason': reason,
             'scope_id': 'layer_ensemble_20260914_core_seed7', 'fixed_epochs': 20,
             'matrix': [[arm, 7] for arm in ('block2', 'block5', 'block8', 'block11')],
             'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
             'test_evaluated': False,
             'missing_base_receipts': [arm for arm in ('block2', 'block5', 'block8', 'block11')
                 if not (root/'runs'/arm/'seed7/complete.json').is_file()],
             'evaluation_report_sha256': file_sha256(report) if report.is_file() else None}
    atomic_json(root/'manifests/terminal.json', state)
    if status != 'complete':
        (root/'REPORT.md').write_text('# September14 core experiment — incomplete\n\n'
            + 'The required complete four-model, fixed20epoch comparison is unavailable.\n\n'
            + 'Reason: '+reason+'\n\nNo subset ensemble or ranking of truncated models is reported. '
            + 'Atomic completed-epoch checkpoints, stage receipts and logs are retained.\n')
    else:
        shutil.copy2(root/'evaluation/REPORT.md', root/'REPORT.md')
    from .control import ledger, state_of
    jobs = ledger(root)
    ids = list(jobs.values())
    if ids:
        accounting = subprocess.run(['sacct', '-j', ','.join(ids + ['892193']), '-P',
            '--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode'], text=True, capture_output=True, timeout=30)
        (root/'manifests/accounting.txt').write_text(accounting.stdout)
    workflow = load_workflow()
    observed = state_of(ids + [workflow['capture_job_id']])
    terminal_states = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}
    allocations = {'capture': (workflow['capture_job_id'], workflow['capture_cap_minutes'])}
    allocations.update({stage: (jobs.get(stage), row['resources']['minutes'])
                        for stage, row in workflow['stages'].items() if row['resources']['gpu']})
    accounting_rows = {}
    actual_seconds = workflow['historical_diagnostic_actual_seconds']
    outstanding_seconds = 0
    for name, (job, minutes) in allocations.items():
        observation = observed.get(job, {})
        elapsed = observation.get('elapsed', '')
        finished = observation.get('state') in terminal_states and elapsed.isdigit()
        actual = int(elapsed) if finished else None
        if actual is not None: actual_seconds += actual
        else: outstanding_seconds += minutes*60
        accounting_rows[name] = {'job_id': job, 'state': observation.get('state', 'not_submitted_or_unknown'),
            'nominal_cap_minutes': minutes, 'actual_seconds_if_terminal': actual,
            'reserved_seconds_if_not_terminal': 0 if finished else minutes*60}
    atomic_json(root/'manifests/gpu_accounting.json', {'actual_terminal_gpu_seconds': actual_seconds,
        'conservative_outstanding_reserved_gpu_seconds': outstanding_seconds,
        'actual_plus_outstanding_reserved_gpu_hours': (actual_seconds+outstanding_seconds)/3600,
        'historical_diagnostics_actual_seconds': workflow['historical_diagnostic_actual_seconds'],
        'historical_diagnostics_nominal_cap_minutes': workflow['historical_diagnostic_cap_minutes'],
        'whole_workflow_nominal_cap_minutes': workflow['planned_gpu_cap_minutes_including_historical_reserves'],
        'budget_gpu_hours': 50, 'allocations': accounting_rows,
        'note': 'One GPU per listed allocation. Unknown or unsubmitted future stages retain their conservative cap; CPU jobs excluded.'})
    bundle = publication/('snapshot_'+os.environ['SLURM_JOB_ID'])
    bundle.mkdir(parents=True, exist_ok=True)
    selected = []
    for directory in ('manifests', 'logs', 'runs', 'predictions', 'heads', 'evaluation'):
        if (root/directory).exists():
            selected += [p for p in (root/directory).rglob('*') if p.is_file()
                         and p.suffix in ('.json', '.txt', '.tsv', '.out', '.err', '.md', '.npz', '.safetensors', '.pt')
                         and '.tmp' not in p.name]
    selected += [p for p in root.iterdir() if p.is_file() and p.suffix in ('.json', '.md')]
    for path in selected:
        destination = bundle/path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    shutil.copytree(args.release, bundle/'source', dirs_exist_ok=True)
    workflow = Path(os.environ['OMRI_WORKFLOW_PATH'])
    shutil.copy2(workflow, bundle/'workflow.json')
    screen = root.parent/'layer_screen_20260913'
    for source in (screen/'ops/workflow_capture_four_20260914.json',
                   screen/'feature_cache/manifest.json', screen/'feature_cache/capture_summary.json'):
        if source.is_file():
            destination = bundle/'shared_capture'/source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    atomic_json(bundle/'backup_manifest.json', {'scope': 'Private source/checkpoints/predictions/report; no raw images, feature tensors or credentials',
        'files': {str(p.relative_to(bundle)): file_sha256(p) for p in bundle.rglob('*')
                  if p.is_file() and p.name != 'backup_manifest.json'}})
    os.environ.pop('HF_HUB_OFFLINE', None)
    receipt = publication/'receipt.json'
    for attempt in range(3):
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            repo = 'omrifahn/polygraph-experiments'
            if not api.repo_info(repo, repo_type='model').private:
                raise RuntimeError('Destination must be private')
            uploaded = api.upload_folder(repo_id=repo, repo_type='model', folder_path=str(bundle),
                path_in_repo='layer_ensemble_20260914/'+bundle.name,
                commit_message='Preserve September14 core20epoch experiment: '+status)
            atomic_json(receipt, {'status': 'uploaded', 'experiment_status': status,
                'commit_oid': uploaded.oid, 'commit_url': uploaded.commit_url,
                'snapshot': str(bundle), 'attempt': attempt+1})
            return state
        except Exception as error:
            atomic_json(receipt, {'status': 'upload_failed', 'experiment_status': status,
                        'error_type': type(error).__name__, 'snapshot': str(bundle), 'attempt': attempt+1})
            if attempt < 2:
                time.sleep(60)
    # Remote course-storage report remains usable when Hub service/network fails.
    return state
