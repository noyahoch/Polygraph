"""Durable four-fit validation summary and private backup, including failures."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .runtime import atomic_json, file_sha256, require_slurm, submit_stage

SCOPE = 'layer_screen_four_fits_20260914'
MATRIX = [('block11', 7), ('union4', 7), ('block11', 17), ('union4', 17)]
FIT_STAGES = {f'fit_four_{arm}_s{seed}' for arm, seed in MATRIX}
FINAL = 'finalize_four_20260914'


def ledger(root):
    found = {}
    path = root/'manifests/submissions.tsv'
    if path.exists():
        for line in path.read_text().splitlines():
            row = line.split('\t')
            if len(row) >= 3 and row[1].isdigit():
                if row[2] in found and found[row[2]] != row[1]:
                    raise RuntimeError('Duplicate stage receipts: ' + row[2])
                found[row[2]] = row[1]
    return found


def guardian(args):
    jobs = ledger(args.root)
    if FINAL in jobs:
        result = {'status': 'finalizer_already_submitted', 'job_id': jobs[FINAL]}
    elif not args.after_ramp and 'ramp_four_20260914' in jobs:
        job = submit_stage(args.root, args.old_root, args.release, 'guardian_after_ramp_four_20260914',
                           5, False, 1, 'afterany:' + jobs['ramp_four_20260914'])
        result = {'status': 'guardian_after_ramp_submitted', 'job_id': job}
    else:
        ids = [job for stage, job in jobs.items() if stage in FIT_STAGES]
        dependency = 'afterany:' + ':'.join(ids) if ids else None
        job = submit_stage(args.root, args.old_root, args.release, FINAL, 90, False, 2, dependency)
        result = {'status': 'failure_finalizer_submitted', 'job_id': job, 'fit_dependencies': ids}
    suffix = 'after_ramp' if args.after_ramp else 'after_dispatch'
    atomic_json(args.root/f'manifests/guardian_four_{suffix}_20260914.json', result)
    print(json.dumps(result), flush=True)


def summarize(root):
    import numpy as np
    from ..train import _verify_complete, auroc
    rows, arrays = {}, {}
    for arm, seed in MATRIX:
        directory = root/'runs_four_20260914'/arm/f'seed{seed}'
        complete = _verify_complete(directory)
        if complete.get('arm') != arm or complete.get('seed') != seed:
            raise RuntimeError('Completed run identity mismatch')
        with np.load(directory/'validation.npz', allow_pickle=False) as z:
            data = {key: z[key].copy() for key in z.files}
        key = f'{arm}/seed{seed}'
        arrays[key] = data
        rows[key] = {'validation_auroc': auroc(data['y'], data['score']),
                     'best_epoch': complete['best_epoch'], 'completed_epochs': complete['completed_epochs'],
                     'validation_sha256': file_sha256(directory/'validation.npz')}
    reference = arrays['block11/seed7']
    for key, data in arrays.items():
        if not all(np.array_equal(reference[field], data[field]) for field in
                   ('record_id', 'image_id', 'source_id', 'severity', 'y', 'label', 'pred')):
            raise RuntimeError('Validation identity mismatch: ' + key)
        if len(data['score']) != 7200 or not np.isfinite(data['score']).all():
            raise RuntimeError('Invalid validation scores: ' + key)
    paired = {str(seed): rows[f'union4/seed{seed}']['validation_auroc'] -
                        rows[f'block11/seed{seed}']['validation_auroc'] for seed in (7, 17)}
    result = {'scope_id': SCOPE, 'development_only': True, 'test_evaluated': False,
              'matrix': MATRIX, 'fits': rows, 'union4_minus_block11_by_seed': paired,
              'mean_paired_validation_auroc_difference': sum(paired.values())/2,
              'caveat': 'Validation selected checkpoints; two paired seeds only. No held-out test claim, no topology-control inference, no new ensemble.'}
    atomic_json(root/'summary_four_20260914.json', result)
    return result


def finalize(args):
    root = args.root
    terminal = root/'manifests/terminal_four_20260914.json'
    receipt = root/'publication/receipt_four_20260914.json'
    root.joinpath('publication').mkdir(parents=True, exist_ok=True)
    with (root/'publication/finalize_four.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists() and json.loads(receipt.read_text()).get('status') == 'uploaded':
            print('Existing completed private backup retained', flush=True)
            return
        admission = json.loads(args.admission.read_text()) if args.admission.exists() else None
        if admission is not None and (admission.get('scope_id') != SCOPE or {tuple(row) for row in admission['matrix']} != set(MATRIX)):
            raise RuntimeError('Unexpected active four-fit scope')
        missing = [f'{arm}/seed{seed}' for arm, seed in MATRIX if not (root/'runs_four_20260914'/arm/f'seed{seed}/complete.json').exists()]
        state = {'scope_id': SCOPE, 'status': 'partial' if missing else 'ready_for_summary',
                 'missing_fits': missing, 'test_evaluated': False, 'job_id': os.environ['SLURM_JOB_ID'],
                 'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                 'admission_sha256': file_sha256(args.admission) if admission is not None else None,
                 'admission_missing': admission is None}
        if not missing and admission is not None:
            try:
                summarize(root)
                state['status'] = 'complete'
            except Exception as error:
                state.update(status='summary_failed', summary_error=repr(error))
        ids = list(ledger(root).values())
        if ids:
            result = subprocess.run(['sacct', '-j', ','.join(ids), '--parsable2', '--noheader',
                '--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode'], text=True, capture_output=True, timeout=30)
            (root/'manifests/accounting_four_20260914.txt').write_text(result.stdout)
            state['accounting_exit_code'] = result.returncode
        atomic_json(terminal, state)
        bundle = root/'publication'/('four_fit_snapshot_' + os.environ['SLURM_JOB_ID'])
        bundle.mkdir(parents=True, exist_ok=True)
        paths = []
        for name in ('manifests', 'logs'):
            if (root/name).exists():
                paths += [p for p in (root/name).rglob('*') if p.is_file() and p.suffix in {'.json', '.txt', '.tsv', '.out', '.err'}]
        for arm, seed in MATRIX:
            directory = root/'runs_four_20260914'/arm/f'seed{seed}'
            if directory.exists():
                paths += [p for p in directory.iterdir() if p.is_file() and p.suffix in {'.json', '.npz', '.safetensors', '.pt'}]
        for name in ('preflight.json', 'summary_four_20260914.json', 'feature_cache/manifest.json', 'feature_cache/capture_summary.json'):
            if (root/name).is_file():
                paths.append(root/name)
        for path in paths:
            target = bundle/path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        shutil.copytree(args.release, bundle/'source', dirs_exist_ok=True)
        workflow_dir = bundle/'workflows'
        workflow_dir.mkdir(exist_ok=True)
        for workflow_path in (Path(os.environ['OMRI_WORKFLOW_PATH']), root/'ops/workflow_capture_four_20260914.json'):
            shutil.copy2(workflow_path, workflow_dir/workflow_path.name)
        manifest = {str(p.relative_to(bundle)): file_sha256(p) for p in bundle.rglob('*') if p.is_file() and p.name != 'backup_manifest.json'}
        atomic_json(bundle/'backup_manifest.json', {'files': manifest, 'scope_id': SCOPE,
            'scope': 'Source, four-run checkpoints and validation, compact manifests/logs. No images, feature tensors, diagnostic state tensors or credentials.'})
        try:
            os.environ.pop('HF_HUB_OFFLINE', None)
            from huggingface_hub import HfApi
            api = HfApi()
            repo = 'omrifahn/polygraph-experiments'
            if not api.repo_info(repo, repo_type='model').private:
                raise RuntimeError('Refusing nonprivate backup destination')
            uploaded = api.upload_folder(repo_id=repo, repo_type='model', folder_path=str(bundle),
                path_in_repo='layer_screen_20260913/' + bundle.name,
                commit_message='Save four-fit September14 development comparison: ' + state['status'])
            atomic_json(receipt, {'status': 'uploaded', 'commit_url': uploaded.commit_url,
                'commit_oid': uploaded.oid, 'snapshot': str(bundle), 'experiment_status': state['status']})
        except Exception as error:
            atomic_json(receipt, {'status': 'upload_failed', 'error': type(error).__name__,
                'local_backup': str(bundle), 'experiment_status': state['status']})
            raise
        print(json.dumps(state), flush=True)


def main():
    require_slurm()
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'old-root', 'release', 'admission'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--guardian', action='store_true')
    p.add_argument('--after-ramp', action='store_true')
    args = p.parse_args()
    if args.guardian:
        guardian(args)
    else:
        finalize(args)


if __name__ == '__main__':
    main()
