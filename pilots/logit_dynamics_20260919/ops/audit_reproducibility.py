"""Read-only Slurm CPU audit of the completed, fixed LogitDynamics artifacts.

Loads portable files freshly fetched from their pinned private Hugging Face commit;
compares native weights exactly, rebuilds train-only statistics and reference
predictions, and independently recomputes saved point metrics. Never fits models.
This operator script is outside the sealed scientific package and release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def verify(path, expected):
    require(sha(path) == expected, 'Checksum mismatch: ' + str(path))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def deadline(_signum, _frame):
    raise TimeoutError('Fixed 25-minute internal audit deadline reached')


def run(args, result):
    # Import numerical libraries only after Slurm and bounded-output guards in main.
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from sklearn.metrics import average_precision_score, roc_auc_score

    torch.set_num_threads(6)
    torch.set_float32_matmul_precision('highest')
    root, release = args.root.resolve(), args.release.resolve()
    source = read(release / 'source_manifest.json')
    for name, expected in source['files'].items():
        verify(release / name, expected)
    sys.path.insert(0, str(release))
    from pilots.logit_dynamics_20260919.protocol import campaign, evaluation_gate, METADATA, SEEDS, RECIPE
    from pilots.logit_dynamics_20260919.data import load_role
    from pilots.logit_dynamics_20260919.features import build_features, fit_normalizer, normalize, feature_names
    from pilots.logit_dynamics_20260919.train import LayerHeads

    contract = campaign(root)
    gate = evaluation_gate(root)
    role_map = read(root / 'role_map.json')['roles']
    index = read(root / 'cls' / 'index.json')
    cls_manifest = read(root / 'cls' / 'manifest.json')
    verify(root / 'cls' / 'index.json', cls_manifest['index_sha256'])
    by_id = {row['record_id']: row for row in index}
    require(len(by_id) == len(index) == 28800, 'CLS record inventory is not exactly 28,800 unique records')
    require(cls_manifest['complete'] is True and cls_manifest['completed_records'] == 28800,
            'Full CLS extraction is incomplete')
    all_photos = set()
    expected_counts = {'head_train': 1200, 'probe_train': 800, 'probe_val': 400, 'dev_eval': 800}
    for role, count in expected_counts.items():
        spec = role_map[role]
        photos = set(spec['photo_ids'])
        require(len(photos) == count and not photos & all_photos, 'Role photograph count/overlap mismatch')
        all_photos |= photos
        rows = [by_id[record] for record in spec['record_ids']]
        require(len(rows) == count * 9 and len(set(spec['record_ids'])) == len(rows), 'Role record count/duplication')
        observed = {photo: 0 for photo in photos}
        for row in rows:
            require(row['image_id'] in photos, 'Role source photograph disagrees with record membership')
            require(row['y'] == int(row['pred'] != row['label']), 'Frozen error target changed')
            observed[row['image_id']] += 1
        require(set(observed.values()) == {9}, 'A source photograph lost one of its nine views')
    result['provenance'] = {
        'campaign_sha256': sha(root / 'campaign.json'), 'gate_sha256': sha(root / 'evaluation_gate.json'),
        'source_identity_sha256': source['source_identity_sha256'], 'cls_manifest_sha256': sha(root / 'cls' / 'manifest.json'),
        'all_source_files_verified': len(source['files']), 'roles_disjoint': True,
        'role_photographs': expected_counts, 'full_extraction_parity_maximum': cls_manifest['parity_maximum'],
    }

    publication = read(root / 'publication' / 'receipt.json')
    require(publication['status'] == 'uploaded' and publication['private'] is True, 'Completed private backup receipt required')
    revision, repo = publication['commit_oid'], publication['repo_id']
    prefix = root.name + '/' + Path(publication['snapshot']).name
    backup_manifest = read(Path(publication['snapshot']) / 'backup_manifest.json')
    verify(Path(publication['snapshot']) / 'backup_manifest.json', publication['manifest_sha256'])
    require(len(revision) == 40, 'Full immutable Hugging Face commit required')
    published = args.published_dir.resolve()
    fetched = read(published / 'FETCH_RECEIPT.json')
    require(fetched['complete'] is True and fetched['revision'] == revision
            and fetched['repo_id'] == repo and fetched['manifest_sha256'] == publication['manifest_sha256'],
            'Fresh published-artifact receipt does not match the pinned backup')
    verify(published / 'backup_manifest.json', publication['manifest_sha256'])
    for relative, specification in fetched['files'].items():
        verify(published / relative, specification['sha256'])
        require(specification['sha256'] == backup_manifest['files'][relative], 'Fetched file is not bound by published manifest')
        original = release / 'source_manifest.json' if relative == 'source/source_manifest.json' else root / relative
        verify(original, specification['sha256'])
    downloaded = {}
    for seed in SEEDS:
        for relative in (f'runs/seed{seed}/heads/model.safetensors', f'runs/seed{seed}/probe/model.safetensors',
                         f'runs/seed{seed}/probe/normalizer.json'):
            local = published / relative
            verify(local, backup_manifest['files'][relative])
            verify(local, sha(root / relative))
            downloaded[relative] = Path(local)
    result['portable_download'] = {'repo_id': repo, 'revision': revision, 'prefix': prefix,
                                   'private_from_publication_receipt': True,
                                   'fresh_published_directory': str(published),
                                   'fetch_receipt_sha256': sha(published / 'FETCH_RECEIPT.json'),
                                   'fetched_files_verified': len(fetched['files']), 'portable_files_loaded': len(downloaded),
                                   'sha256': {name: sha(path) for name, path in downloaded.items()}}

    def metadata_equal(archive, role):
        rows = [by_id[record] for record in role_map[role]['record_ids']]
        for key in METADATA:
            expected = np.asarray([row[key] for row in rows], dtype=np.int64)
            require(archive[key].dtype == np.int64 and np.array_equal(archive[key], expected),
                    'Saved NPZ metadata differs from frozen CLS/index role: ' + role + '/' + key)

    def drift(actual, expected, description):
        actual, expected = np.asarray(actual), np.asarray(expected)
        require(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(),
                'Nonfinite/misaligned replay: ' + description)
        difference = np.abs(actual - expected)
        passed = bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-4))
        detail = {'passed': passed, 'maximum_absolute_difference': float(difference.max()),
                  'atol': 1e-4, 'rtol': 1e-4, 'values': int(actual.size)}
        result.setdefault('numerical_replay', {})[description] = detail
        require(passed, 'Predeclared CPU replay tolerance exceeded: ' + description)

    def features_for(model, role_data):
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(role_data['cls']), 512):
                outputs.append(model(role_data['cls'][start:start + 512].float()).numpy())
        return build_features(np.concatenate(outputs), role_data['logits'].numpy(), layers=12, k=5)

    result['seeds'] = {}
    portable_models = {}
    for seed in SEEDS:
        directory = root / 'runs' / f'seed{seed}'
        config = read(directory / 'probe' / 'config.json')
        head_config = read(directory / 'heads' / 'config.json')
        complete = read(directory / 'probe' / 'complete.json')
        head_complete = read(directory / 'heads' / 'complete.json')
        heads_history = read(directory / 'heads' / 'history.json')
        history = read(directory / 'probe' / 'history.json')
        require([row['epoch'] for row in heads_history] == list(range(1, 17)), 'Head epoch inventory changed')
        require([row['epoch'] for row in history] == list(range(1, 101)), 'Probe epoch inventory changed')
        require(head_complete['epochs'] == 16 and head_complete['selected_epoch'] == 16, 'Heads must use fixed epoch16')
        require(complete['epochs'] == 100, 'Probe must finish all100 epochs')
        require(head_config['recipe'] == RECIPE['head'] and config['recipe'] == RECIPE['probe'], 'Optimizer/budget contract changed')
        require(config['features'] == feature_names(12, 5), 'Named feature order changed')
        require(config['training_role'] == 'probe_train' and config['selection_role'] == 'probe_val'
                and config['selection_metric'] == 'average_precision', 'Probe training/selection roles changed')
        for row in heads_history:
            require(len(row['cross_entropy_per_layer']) == 12
                    and all(math.isfinite(v) and v >= 0 for v in row['cross_entropy_per_layer']), 'Invalid head loss history')
        best, chosen = -1.0, None
        for row in history:
            value = row['validation_average_precision']
            require(math.isfinite(value) and 0 <= value <= 1 and math.isfinite(row['training_loss'])
                    and row['training_loss'] >= 0, 'Invalid probe history')
            if value > best:
                best, chosen = value, row['epoch']
            require(row['best_epoch'] == chosen and row['best_average_precision'] == best, 'Historical earliest-best selection changed')
        require(complete['selected_epoch'] == chosen and complete['validation_average_precision'] == best,
                'Final checkpoint differs from strict earliest AP maximum')
        states = {}
        for stage in ('heads', 'probe'):
            receipt = read(directory / stage / 'complete.json')
            for name, expected in receipt['files'].items():
                verify(directory / stage / name, expected)
            native = torch.load(directory / stage / 'checkpoint.pt', map_location='cpu', weights_only=True)
            portable = load_file(str(downloaded[f'runs/seed{seed}/{stage}/model.safetensors']), device='cpu')
            require(native['epoch'] == (16 if stage == 'heads' else chosen), 'Native checkpoint epoch differs')
            require(native['identity'] == receipt['identity'], 'Native checkpoint identity differs')
            require(set(native['state_dict']) == set(portable), 'Portable weight-key inventory differs')
            for key, tensor in portable.items():
                require(tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
                        and torch.equal(tensor, native['state_dict'][key]), 'Portable/native tensor mismatch: ' + key)
            states[stage] = portable
        heads, probe = LayerHeads().eval(), torch.nn.Linear(85, 1).eval()
        heads.load_state_dict(states['heads']); probe.load_state_dict(states['probe'])
        heads.requires_grad_(False); probe.requires_grad_(False)
        normalizer = read(downloaded[f'runs/seed{seed}/probe/normalizer.json'])
        require(normalizer['fit_role'] == 'probe_train' and normalizer['records'] == 7200 and normalizer['ddof'] == 0,
                'Scaler role/count/population convention changed')
        verify(downloaded[f'runs/seed{seed}/probe/normalizer.json'], config['normalizer_sha256'])
        errors = sum(by_id[record]['y'] for record in role_map['probe_train']['record_ids'])
        require(config['positive_weight'] == (7200 - errors) / errors, 'Positive weight is not derived from probe_train')
        with np.load(directory / 'probe' / 'validation.npz', allow_pickle=False) as archive:
            metadata_equal(archive, 'probe_val')
            require(float(average_precision_score(archive['y'], archive['score'])) == best,
                    'Saved validation scores do not reproduce selected AP')
        portable_models[seed] = (heads, probe, normalizer)
        result['seeds'][str(seed)] = {'head_epochs': 16, 'probe_epochs': 100, 'selected_epoch': chosen,
            'selected_validation_average_precision': best, 'earliest_tie_rule_verified': True,
            'native_and_portable_weights_exact': True, 'positive_weight': config['positive_weight'],
            'scaler_fit_role': normalizer['fit_role'], 'scaler_records': normalizer['records']}

    # Each role is loaded once and shared across seeds; all numerical replay runs on CPU.
    for role in ('probe_train', 'probe_val', 'dev_eval'):
        data = load_role(root, role)
        for seed, (heads, probe, normalizer) in portable_models.items():
            raw = features_for(heads, data)
            if role == 'probe_train':
                recomputed = fit_normalizer(raw)
                drift(recomputed['mean'], normalizer['mean'], f'seed{seed}/probe_train_mean')
                drift(recomputed['scale'], normalizer['scale'], f'seed{seed}/probe_train_scale')
            else:
                x = torch.from_numpy(normalize(raw, normalizer))
                with torch.inference_mode():
                    scores = torch.cat([probe(x[start:start + 512]).flatten() for start in range(0, len(x), 512)]).numpy()
                relative = 'probe/validation.npz' if role == 'probe_val' else 'predictions/dev_eval.npz'
                with np.load(root / 'runs' / f'seed{seed}' / relative, allow_pickle=False) as archive:
                    metadata_equal(archive, role)
                    drift(scores, archive['score'], f'seed{seed}/{role}_scores')
            del raw
        del data
        print(json.dumps({'audit_role_replay_complete': role}), flush=True)

    evaluation = root / 'evaluation'
    receipt = read(evaluation / 'complete.json')
    require(receipt['complete'] is True, 'Evaluation not complete')
    for name, expected in receipt['files'].items():
        verify(evaluation / name, expected)
    report = read(evaluation / 'report.json')
    with np.load(evaluation / 'scores.npz', allow_pickle=False) as archive:
        metadata_equal(archive, 'dev_eval')
        keys, scores, labels = archive['score_keys'].tolist(), archive['scores'], archive['y']
        require(scores.shape == (7200, 14) and len(set(keys)) == 14, 'Final score schema changed')
        per_key = {}
        for column, key in enumerate(keys):
            auc, ap = float(roc_auc_score(labels, scores[:, column])), float(average_precision_score(labels, scores[:, column]))
            saved = report['per_seed_metrics'][key]
            require(abs(saved['auroc'] - auc) <= 1e-12 and abs(saved['average_precision'] - ap) <= 1e-12,
                    'Saved point metric differs from independent sklearn recomputation: ' + key)
            per_key[key] = {'auroc': auc, 'average_precision': ap}
            if key.startswith('LogitDynamics/seed'):
                seed = key.split('seed')[-1]
                with np.load(root / 'runs' / ('seed' + seed) / 'predictions' / 'dev_eval.npz', allow_pickle=False) as prediction:
                    require(np.array_equal(scores[:, column], prediction['score']), 'Reported LD scores differ from saved prediction')
        actual_delta = sum(per_key[f'G_mean/seed{s}']['auroc'] - per_key[f'LogitDynamics/seed{s}']['auroc'] for s in SEEDS) / 3
        require(abs(actual_delta - report['primary']['estimate']) <= 1e-12, 'Primary point estimate changed')
        require(report['error_count'] == int(labels.sum()) and report['error_prevalence'] == float(labels.mean()), 'Error prevalence changed')
        baseline_path = Path(contract['baseline_root']) / 'evaluation' / 'scores.npz'
        verify(baseline_path, contract['baseline_scores_sha256'])
        with np.load(baseline_path, allow_pickle=False) as baseline:
            metadata_equal(baseline, 'dev_eval')
            original = dict(zip(baseline['score_keys'].tolist(), baseline['scores'].T))
            for column, key in enumerate(keys):
                if not key.startswith('LogitDynamics/'):
                    require(np.array_equal(scores[:, column], original[key]), 'Historical score vector changed: ' + key)
        result['point_metrics'] = {'all_14_vectors_auroc_ap_recomputed': True, 'metadata_exact': True,
                                  'historical_vectors_exact': True, 'primary_delta': actual_delta, 'per_vector': per_key}
    with np.load(evaluation / 'bootstrap.npz', allow_pickle=False) as archive:
        require(archive['multiplicity'].shape == (2000, 800), 'Bootstrap multiplicity shape changed')
        require(np.all(archive['multiplicity'] >= 0) and np.all(archive['multiplicity'].sum(axis=1) == 800),
                'Bootstrap source draws are not complete')
        require(archive['per_seed_differences'].shape == (2000, 3), 'Bootstrap seed matrix shape changed')
        require(np.array_equal(archive['mean_differences'], archive['per_seed_differences'].mean(axis=1), equal_nan=True),
                'Bootstrap draw means do not average within-seed differences')
        undefined = np.flatnonzero(~np.isfinite(archive['mean_differences'])).tolist()
        require(undefined == report['primary']['undefined_draw_indices'], 'Undefined draw inventory changed')
        ci = None if undefined else np.quantile(archive['mean_differences'], [.025, .975], method='linear').tolist()
        require(ci == report['primary']['interval_95'], 'Saved interval differs from stored-draw percentiles')
        result['bootstrap_artifact'] = {'draws': 2000, 'sources': 800, 'seed_columns': 3,
            'percentiles_and_seed_means_verified': True, 'weighted_auroc_per_draw_recomputed': False,
            'scope': 'Stored bootstrap structural/aggregation audit; mathematical paired-AUROC tests passed before fitting'}
    result['complete'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--published-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    require(bool(os.environ.get('SLURM_JOB_ID')), 'Audit numerical work is Slurm-only')
    require(not args.out.exists(), 'Existing audit receipt must not be overwritten')
    require(not os.environ.get('CUDA_VISIBLE_DEVICES'), 'Set CUDA_VISIBLE_DEVICES empty for this CPU-only audit')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(25 * 60)
    started = time.monotonic()
    result = {'complete': False, 'job_id': os.environ['SLURM_JOB_ID'], 'audit_script_sha256': sha(__file__),
              'scope': 'Read-only completed-artifact, portable-checkpoint and CPU inference audit; no fitting',
              'CPU_replay_tolerance': {'atol': 1e-4, 'rtol': 1e-4}, 'internal_limit_seconds': 1500}
    try:
        run(args, result)
    except BaseException as error:
        result['error_type'] = type(error).__name__
        result['error'] = str(error)
        raise
    finally:
        signal.alarm(0)
        result['elapsed_seconds'] = time.monotonic() - started
        write(args.out, result)
        print(json.dumps({'complete': result['complete'], 'receipt': str(args.out), 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
