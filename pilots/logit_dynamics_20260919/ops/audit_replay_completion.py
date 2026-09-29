"""Complete cached-CLS portable replay without clearing earlier failed audits.

Run only in a Slurm allocation. CPU and CUDA modes each replay every validation
and development row for all three seeds from pinned published safetensors.
CPU additionally recomputes every paired bootstrap draw using scikit-learn.
Score drift is recorded for all rows at the original fixed tolerance; malformed
inputs, nonfinite arithmetic and provenance failures remain fatal. No fitting,
precision alternatives, scientific-output replacement or raw-image extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time


REVISION = '86605781c0528e286483e305072f7787393da860'
SOURCE_SHA = 'd78438ba6a0c512ab66e5b59a16a772b0c4d972e39e9bf57ceb043b8936144b3'
CAMPAIGN_SHA = 'f133671b92871129a0e859e47188c316e8ef535634a1fda33a9efa287dfa094f'
GATE_SHA = '8918ff80c58a74e53a747fb0c054fe18b93e0f3dd06d64ebb1ff1d42741c4ab0'
BACKUP_SHA = 'f2e17250d314568bf7b5815b5eea7d79720e8c9bb24d35289f58530f974ed572'
ATOL = RTOL = 1e-4


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def verify(path, expected):
    require(sha(path) == expected, 'Checksum mismatch: ' + str(path))


def write(path, value):
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def deadline(_signal, _frame):
    raise TimeoutError('Fixed bounded replay deadline reached')


def run(args, result, artifact_dir):
    # Numerical imports occur only after Slurm, output and resource guards.
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from sklearn.metrics import average_precision_score, roc_auc_score
    import sklearn

    root, release, published = args.root.resolve(), args.release.resolve(), args.published_dir.resolve()
    verify(release / 'source_manifest.json', SOURCE_SHA)
    source = read(release / 'source_manifest.json')
    for name, expected in source['files'].items():
        verify(release / name, expected)
    sys.path.insert(0, str(release))
    from pilots.logit_dynamics_20260919.protocol import campaign, evaluation_gate, METADATA, SEEDS
    from pilots.logit_dynamics_20260919.data import load_role
    from pilots.logit_dynamics_20260919.features import normalize
    from pilots.logit_dynamics_20260919.train import LayerHeads, role_features, probe_scores, initialize

    require(tuple(SEEDS) == (7, 17, 27), 'Frozen seed inventory changed')
    verify(root / 'campaign.json', CAMPAIGN_SHA)
    verify(root / 'evaluation_gate.json', GATE_SHA)
    contract = campaign(root)
    evaluation_gate(root)
    publication = read(root / 'publication' / 'receipt.json')
    require(publication['commit_oid'] == REVISION and publication['manifest_sha256'] == BACKUP_SHA
            and publication['repo_id'] == 'omrifahn/polygraph-experiments'
            and publication['private'] is True and publication['status'] == 'uploaded',
            'Unexpected original publication identity')
    verify(published / 'backup_manifest.json', BACKUP_SHA)
    manifest = read(published / 'backup_manifest.json')
    fetched = read(published / 'FETCH_RECEIPT.json')
    require(fetched['complete'] is True and fetched['revision'] == REVISION
            and fetched['repo_id'] == publication['repo_id'] and fetched['manifest_sha256'] == BACKUP_SHA,
            'Pinned published download receipt changed')
    for relative, entry in fetched['files'].items():
        require(entry['sha256'] == manifest['files'][relative], 'Published file manifest mismatch')
        verify(published / relative, entry['sha256'])
    result['provenance'] = {
        'source_manifest_sha256': SOURCE_SHA, 'source_files_verified': len(source['files']),
        'campaign_sha256': CAMPAIGN_SHA, 'evaluation_gate_sha256': GATE_SHA,
        'published_revision': REVISION, 'published_manifest_sha256': BACKUP_SHA,
        'fetched_files_verified': len(fetched['files']),
        'original_failed_audit_sha256': sha(root / 'audit' / 'reproducibility_cpu_v1.json'),
        'original_failed_audit_preserved': True,
    }
    original = read(root / 'audit' / 'reproducibility_cpu_v1.json')
    require(original['complete'] is False and original['error'] ==
            'Predeclared CPU replay tolerance exceeded: seed17/probe_val_scores',
            'Expected historical failure must remain preserved')

    torch.set_num_threads(6)
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.backend)
    if args.backend == 'cuda':
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one allocated CUDA device required')
        initialize(7)
    result['runtime'] = {
        'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
        'numpy': np.__version__, 'sklearn': sklearn.__version__, 'cuda_version': torch.version.cuda,
        'backend': args.backend, 'cpu_threads': torch.get_num_threads(), 'batch_size': 512,
        'matmul_precision': torch.get_float32_matmul_precision(),
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'GPU': torch.cuda.get_device_name(0) if args.backend == 'cuda' else None,
        'existing_environment': True, 'clean_install_tested': False,
    }
    portable = {}
    for seed in SEEDS:
        if args.backend == 'cuda':
            initialize(seed)
        heads = LayerHeads().eval().requires_grad_(False)
        probe = torch.nn.Linear(85, 1).eval().requires_grad_(False)
        for stage, model in (('heads', heads), ('probe', probe)):
            relative = f'runs/seed{seed}/{stage}/model.safetensors'
            verify(published / relative, manifest['files'][relative])
            verify(root / relative, manifest['files'][relative])
            state = load_file(str(published / relative), device='cpu')
            native = torch.load(root / f'runs/seed{seed}/{stage}/checkpoint.pt', map_location='cpu', weights_only=True)
            require(set(state) == set(native['state_dict']), 'Portable/native parameter names changed')
            for name, value in state.items():
                require(value.dtype == torch.float32 and torch.isfinite(value).all()
                        and torch.equal(value, native['state_dict'][name]), 'Portable/native parameter mismatch')
            model.load_state_dict(state)
            model.to(device)
        relative = f'runs/seed{seed}/probe/normalizer.json'
        verify(published / relative, manifest['files'][relative])
        scaler = read(published / relative)
        require(scaler['fit_role'] == 'probe_train' and scaler['records'] == 7200 and scaler['ddof'] == 0,
                'Frozen training scaler identity changed')
        portable[seed] = (heads, probe, scaler)
    result['native_portable_parameter_equality'] = True
    result['replay'] = {}

    # Reuse the exact frozen production feature and probe batching routines on
    # either requested backend. No score mismatch is allowed to skip other seeds.
    for role, records in (('probe_val', 3600), ('dev_eval', 7200)):
        data = load_role(root, role)
        require(len(data['cls']) == records, 'Wrong frozen role size')
        for seed in SEEDS:
            if args.backend == 'cuda':
                initialize(seed)
            heads, probe, scaler = portable[seed]
            raw = role_features(heads, data, device)
            values = torch.from_numpy(normalize(raw, scaler))
            actual = probe_scores(probe, values, device, batch_size=512)
            relative = f'runs/seed{seed}/' + ('probe/validation.npz' if role == 'probe_val' else 'predictions/dev_eval.npz')
            verify(published / relative, manifest['files'][relative])
            verify(root / relative, manifest['files'][relative])
            with np.load(published / relative, allow_pickle=False) as saved:
                for key in METADATA:
                    require(saved[key].dtype == np.int64 and np.array_equal(saved[key], data['metadata'][key]),
                            'Saved/replayed metadata mismatch: ' + key)
                expected = saved['score'].copy()
            require(actual.shape == expected.shape == (records,) and np.isfinite(actual).all()
                    and np.isfinite(expected).all(), 'Malformed or nonfinite replay')
            affected = np.flatnonzero(~np.isclose(actual, expected, atol=ATOL, rtol=RTOL))
            y = data['metadata']['y']
            key = f'seed{seed}/{role}'
            evidence = artifact_dir / f'seed{seed}_{role}.npz'
            with evidence.open('xb') as stream:
                np.savez_compressed(stream, **data['metadata'], replay_score=actual, saved_score=expected)
            result['replay'][key] = {
                'tolerance_passed': len(affected) == 0, 'records': records,
                'violation_count': int(len(affected)), 'atol': ATOL, 'rtol': RTOL,
                'maximum_absolute_difference': float(np.max(np.abs(actual - expected))),
                'bitwise_scores_equal': bool(np.array_equal(actual, expected)),
                'saved_metrics': {'auroc': float(roc_auc_score(y, expected)), 'average_precision': float(average_precision_score(y, expected))},
                'replay_metrics': {'auroc': float(roc_auc_score(y, actual)), 'average_precision': float(average_precision_score(y, actual))},
                'violations': [{'row_index': int(i), 'record_id': int(data['metadata']['record_id'][i]),
                                'image_id': int(data['metadata']['image_id'][i]),
                                'saved_score': float(expected[i]), 'replay_score': float(actual[i]),
                                'absolute_difference': float(abs(actual[i] - expected[i]))} for i in affected],
                'score_artifact': str(evidence), 'score_artifact_sha256': sha(evidence),
            }
            write(args.out, result)
            print(json.dumps({'replayed': key, 'backend': args.backend, 'violations': int(len(affected))}), flush=True)
            del raw, values
        del data
    result['all_six_replays_executed'] = len(result['replay']) == 6
    result['tolerance_passed'] = all(value['tolerance_passed'] for value in result['replay'].values())

    if args.backend == 'cpu':
        evaluation = root / 'evaluation'
        complete = read(evaluation / 'complete.json')
        require(complete['complete'] is True, 'Original evaluation is incomplete')
        for relative, expected in complete['files'].items():
            verify(evaluation / relative, expected)
        report = read(evaluation / 'report.json')
        with np.load(evaluation / 'scores.npz', allow_pickle=False) as archive:
            scores, labels = archive['scores'].copy(), archive['y'].copy()
            keys, photos = archive['score_keys'].tolist(), archive['image_id'].copy()
        require(scores.shape == (7200, 14) and np.isfinite(scores).all(), 'Malformed saved score matrix')
        points = {}
        for column, key in enumerate(keys):
            auc = float(roc_auc_score(labels, scores[:, column]))
            ap = float(average_precision_score(labels, scores[:, column]))
            require(abs(auc - report['per_seed_metrics'][key]['auroc']) <= 1e-12
                    and abs(ap - report['per_seed_metrics'][key]['average_precision']) <= 1e-12,
                    'Independently recalculated saved metric differs: ' + key)
            points[key] = {'auroc': auc, 'average_precision': ap}
        result['independent_point_metrics'] = points
        groups, membership, counts = np.unique(photos, return_inverse=True, return_counts=True)
        require(len(groups) == 800 and np.all(counts == 9), 'Source photo grouping changed')
        with np.load(evaluation / 'bootstrap.npz', allow_pickle=False) as archive:
            require(np.array_equal(archive['image_id'], groups), 'Bootstrap photo order changed')
            multiplicity, original_deltas = archive['multiplicity'].copy(), archive['per_seed_differences'].copy()
            original_means = archive['mean_differences'].copy()
        require(multiplicity.shape == (2000, 800) and original_deltas.shape == (2000, 3), 'Bootstrap dimensions changed')
        require(np.issubdtype(multiplicity.dtype, np.integer) and np.all(multiplicity >= 0)
                and np.all(multiplicity.sum(axis=1) == 800), 'Malformed source multiplicities')
        rng = np.random.default_rng(20260919)
        deltas = np.full((2000, 3), np.nan, dtype=np.float64)
        for draw in range(2000):
            counts = np.bincount(rng.integers(0, 800, size=800), minlength=800)
            require(np.array_equal(counts, multiplicity[draw]), 'Seeded paired-photo draw differs')
            weights = counts[membership]
            if weights[labels == 0].sum() > 0 and weights[labels == 1].sum() > 0:
                for column, seed in enumerate(SEEDS):
                    left = roc_auc_score(labels, scores[:, keys.index(f'G_mean/seed{seed}')], sample_weight=weights)
                    right = roc_auc_score(labels, scores[:, keys.index(f'LogitDynamics/seed{seed}')], sample_weight=weights)
                    deltas[draw, column] = left - right
            if (draw + 1) % 250 == 0:
                print(json.dumps({'independent_sklearn_bootstrap_draws': draw + 1}), flush=True)
        undefined = np.flatnonzero(~np.isfinite(deltas.mean(axis=1))).tolist()
        require(undefined == report['primary']['undefined_draw_indices'], 'Undefined draw inventory changed')
        require(np.allclose(deltas, original_deltas, atol=1e-12, rtol=0, equal_nan=True),
                'Independently recomputed bootstrap seed differences disagree')
        means = deltas.mean(axis=1)
        require(np.allclose(means, original_means, atol=1e-12, rtol=0, equal_nan=True), 'Bootstrap draw means disagree')
        interval = None if undefined else np.quantile(means, [.025, .975], method='linear').tolist()
        require((interval is None and report['primary']['interval_95'] is None) or
                (interval is not None and np.allclose(interval, report['primary']['interval_95'], atol=1e-12, rtol=0)),
                'Independent bootstrap confidence interval differs')
        evidence = artifact_dir / 'independent_bootstrap.npz'
        with evidence.open('xb') as stream:
            np.savez_compressed(stream, image_id=groups, multiplicity=multiplicity,
                                per_seed_differences=deltas, mean_differences=means)
        result['independent_bootstrap'] = {
            'passed': True, 'draws': 2000, 'seeds': list(SEEDS),
            'implementation': 'sklearn.metrics.roc_auc_score(sample_weight=shared_photo_multiplicity)',
            'production_WeightedAUC_used': False, 'all_seeded_source_draws_exact': True,
            'weighted_auroc_calculations': 2 * 3 * (2000 - len(undefined)),
            'maximum_seed_difference_vs_saved': float(np.nanmax(np.abs(deltas - original_deltas))),
            'interval_95': interval, 'undefined_draw_indices': undefined, 'comparison_atol': 1e-12,
            'artifact': str(evidence), 'artifact_sha256': sha(evidence),
        }
    else:
        result['independent_bootstrap'] = {'executed': False, 'reason': 'CPU-only companion audit owns independent statistics'}
    result['execution_complete'] = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--published-dir', type=Path, required=True)
    p.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    require(bool(os.environ.get('SLURM_JOB_ID')), 'All numerical work requires a Slurm allocation')
    require(args.out.resolve().parent == args.root.resolve() / 'audit', 'Receipt must be a new direct audit/ child')
    require(not args.out.exists(), 'Existing audit receipt must not be overwritten')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Use the frozen production CUBLAS workspace setting')
    if args.backend == 'cpu':
        require(not os.environ.get('CUDA_VISIBLE_DEVICES'), 'CPU allocation must expose no CUDA device')
    artifact_dir = args.out.with_suffix('')
    require(not artifact_dir.exists(), 'Existing audit evidence must not be overwritten')
    artifact_dir.mkdir(parents=True)
    limit = (19 if args.backend == 'cpu' else 14) * 60
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(limit)
    started = time.monotonic()
    result = {
        'execution_complete': False, 'tolerance_passed': None,
        'job_id': os.environ['SLURM_JOB_ID'], 'script_sha256': sha(__file__),
        'scope': 'Fixed published-checkpoint cached-CLS replay and separate CPU statistics; no fitting',
        'backend': args.backend, 'internal_deadline_seconds': limit,
        'score_tolerance': {'atol': ATOL, 'rtol': RTOL},
        'scientific_results_replaced': False, 'prior_failed_audits_cleared': False,
        'raw_image_end_to_end_tested': False, 'clean_install_tested': False,
    }
    try:
        run(args, result, artifact_dir)
    except BaseException as error:
        result['fatal_error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        signal.alarm(0)
        result['elapsed_seconds'] = time.monotonic() - started
        write(args.out, result)
        print(json.dumps({'execution_complete': result['execution_complete'],
                          'tolerance_passed': result['tolerance_passed'], 'receipt': str(args.out)}), flush=True)


if __name__ == '__main__':
    main()
