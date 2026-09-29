"""Bounded CPU-only diagnosis of the preserved seed17 validation replay failure.

No fitting, no tolerance changes, no writes to scientific artifacts. Complete
saved-score/metric checks first, then inspect only the affected validation rows
with higher-precision head arithmetic. Historical GPU intermediates were not
saved, so a CPU-only diagnostic cannot establish their exact class rankings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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


def deadline(_signal, _frame):
    raise TimeoutError('Fixed nine-minute internal diagnostic deadline reached')


def run(args, result):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from sklearn.metrics import average_precision_score, roc_auc_score
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision('highest')
    root, release, published = args.root.resolve(), args.release.resolve(), args.published_dir.resolve()
    sys.path.insert(0, str(release))
    from pilots.logit_dynamics_20260919.protocol import campaign, evaluation_gate, METADATA, SEEDS
    from pilots.logit_dynamics_20260919.features import build_features, normalize, feature_names
    from pilots.logit_dynamics_20260919.data import load_role
    from pilots.logit_dynamics_20260919.train import LayerHeads

    source = read(release / 'source_manifest.json')
    for name, expected in source['files'].items():
        verify(release / name, expected)
    contract = campaign(root)
    evaluation_gate(root)
    previous = read(root / 'audit' / 'reproducibility_cpu_v1.json')
    require(previous['complete'] is False and previous['error'] ==
            'Predeclared CPU replay tolerance exceeded: seed17/probe_val_scores', 'Expected preserved CPU audit failure required')
    result['prior_audit_sha256'] = sha(root / 'audit' / 'reproducibility_cpu_v1.json')
    fetched = read(published / 'FETCH_RECEIPT.json')
    require(fetched['complete'] is True and fetched['revision'] == '86605781c0528e286483e305072f7787393da860',
            'Exact published snapshot receipt required')
    for name, spec in fetched['files'].items():
        verify(published / name, spec['sha256'])
    index = read(root / 'cls' / 'index.json')
    verify(root / 'cls' / 'index.json', read(root / 'cls' / 'manifest.json')['index_sha256'])
    by_id = {row['record_id']: row for row in index}
    roles = read(root / 'role_map.json')['roles']

    def align(archive, role):
        rows = [by_id[record] for record in roles[role]['record_ids']]
        for key in METADATA:
            wanted = np.asarray([row[key] for row in rows], dtype=np.int64)
            require(archive[key].dtype == np.int64 and np.array_equal(archive[key], wanted),
                    'Metadata misalignment: ' + role + '/' + key)

    evaluation = root / 'evaluation'
    receipt = read(evaluation / 'complete.json')
    require(receipt['complete'] is True, 'Completed scientific evaluation required')
    for name, expected in receipt['files'].items():
        verify(evaluation / name, expected)
    report = read(evaluation / 'report.json')
    result['saved_scores'] = {}
    with np.load(evaluation / 'scores.npz', allow_pickle=False) as archive:
        align(archive, 'dev_eval')
        labels, scores, keys = archive['y'], archive['scores'], archive['score_keys'].tolist()
        require(scores.shape == (7200, 14) and len(set(keys)) == 14, 'Final score matrix shape changed')
        expected_keys = [f'{method}/seed{seed}' for method in ('G_mean', 'S_mean', 'O', 'LogitDynamics') for seed in SEEDS] + ['MSP', 'entropy']
        require(set(keys) == set(expected_keys), 'Final score-key inventory changed')
        point = {}
        for column, key in enumerate(keys):
            auc, ap = float(roc_auc_score(labels, scores[:, column])), float(average_precision_score(labels, scores[:, column]))
            saved = report['per_seed_metrics'][key]
            require(abs(auc - saved['auroc']) <= 1e-12 and abs(ap - saved['average_precision']) <= 1e-12,
                    'Saved AUROC/AP disagrees with independent sklearn: ' + key)
            point[key] = {'auroc': auc, 'average_precision': ap}
        for method in ('G_mean', 'S_mean', 'O', 'LogitDynamics', 'MSP', 'entropy'):
            method_keys = [method] if method in ('MSP', 'entropy') else [f'{method}/seed{s}' for s in SEEDS]
            for metric in ('auroc', 'average_precision'):
                values = [point[key][metric] for key in method_keys]
                summary = report['method_summary'][method][metric]
                require(abs(float(np.mean(values)) - summary['mean']) <= 1e-12, 'Summary mean changed')
                if len(values) > 1:
                    require(abs(float(np.std(values, ddof=1)) - summary['sd']) <= 1e-12, 'Sample seed SD changed')
                else:
                    require(summary['sd'] is None, 'Static score must not have fabricated seed SD')
        for seed in SEEDS:
            for role, relative in (('probe_val', 'probe/validation.npz'), ('dev_eval', 'predictions/dev_eval.npz')):
                path = root / 'runs' / f'seed{seed}' / relative
                with np.load(path, allow_pickle=False) as prediction:
                    align(prediction, role)
                    require(np.isfinite(prediction['score']).all(), 'Nonfinite saved scores')
                    if role == 'dev_eval':
                        require(np.array_equal(scores[:, keys.index(f'LogitDynamics/seed{seed}')], prediction['score']),
                                'Published method scores differ from saved per-seed prediction')
        baseline_path = Path(contract['baseline_root']) / 'evaluation' / 'scores.npz'
        verify(baseline_path, contract['baseline_scores_sha256'])
        with np.load(baseline_path, allow_pickle=False) as baseline:
            align(baseline, 'dev_eval')
            original = dict(zip(baseline['score_keys'].tolist(), baseline['scores'].T))
            for column, key in enumerate(keys):
                if not key.startswith('LogitDynamics/'):
                    require(np.array_equal(scores[:, column], original[key]), 'Historical scores changed: ' + key)
        delta = sum(point[f'G_mean/seed{s}']['auroc'] - point[f'LogitDynamics/seed{s}']['auroc'] for s in SEEDS) / 3
        require(abs(delta - report['primary']['estimate']) <= 1e-12, 'Primary delta changed')
        require(report['error_count'] == int(labels.sum()) and report['error_prevalence'] == float(labels.mean()), 'Prevalence changed')
        result['saved_scores'] = {'passed': True, 'metadata_exact': True, 'historical_scores_exact': True,
                                  'all_14_auroc_ap_and_seed_summary_checks': True, 'primary_delta': delta,
                                  'per_vector': point}
    with np.load(evaluation / 'bootstrap.npz', allow_pickle=False) as bootstrap:
        require(bootstrap['multiplicity'].shape == (2000, 800), 'Bootstrap source-count shape changed')
        require(np.all(bootstrap['multiplicity'] >= 0) and np.all(bootstrap['multiplicity'].sum(axis=1) == 800),
                'Bootstrap source draws malformed')
        require(bootstrap['per_seed_differences'].shape == (2000, 3), 'Bootstrap seed-difference shape changed')
        require(np.array_equal(bootstrap['mean_differences'], bootstrap['per_seed_differences'].mean(axis=1), equal_nan=True),
                'Bootstrap means must average the three within-seed differences')
        undefined = np.flatnonzero(~np.isfinite(bootstrap['mean_differences'])).tolist()
        require(undefined == report['primary']['undefined_draw_indices'], 'Undefined bootstrap inventory changed')
        ci = None if undefined else np.quantile(bootstrap['mean_differences'], [.025, .975], method='linear').tolist()
        require(ci == report['primary']['interval_95'], 'Stored-draw percentile interval changed')
        result['saved_bootstrap'] = {'passed': True, 'interval_95': ci, 'undefined_draws': undefined,
                                    'draws': 2000, 'group': 'image_id', 'weighted_auroc_draws_recomputed': False}
    print(json.dumps({'saved_metadata_metrics_bootstrap_checks': 'passed'}), flush=True)

    # One seed and validation role only; no fitting and no broad replay.
    heads = LayerHeads().eval().requires_grad_(False)
    probe = torch.nn.Linear(85, 1).eval().requires_grad_(False)
    heads.load_state_dict(load_file(str(published / 'runs/seed17/heads/model.safetensors'), device='cpu'))
    probe.load_state_dict(load_file(str(published / 'runs/seed17/probe/model.safetensors'), device='cpu'))
    normalizer = read(published / 'runs/seed17/probe/normalizer.json')
    data = load_role(root, 'probe_val')
    with torch.inference_mode():
        logits32 = torch.cat([heads(data['cls'][start:start + 512].float()) for start in range(0, len(data['cls']), 512)]).numpy()
    features32 = build_features(logits32, data['logits'].numpy(), layers=12, k=5)
    x32 = normalize(features32, normalizer)
    with torch.inference_mode():
        cpu32 = torch.cat([probe(torch.from_numpy(x32[start:start + 512])).flatten() for start in range(0, len(x32), 512)]).numpy()
    with np.load(published / 'runs/seed17/probe/validation.npz', allow_pickle=False) as reference:
        align(reference, 'probe_val')
        saved = reference['score']
    difference = np.abs(cpu32 - saved)
    affected = np.flatnonzero(~np.isclose(cpu32, saved, atol=1e-4, rtol=1e-4))
    result['diagnosis'] = {'seed': 17, 'role': 'probe_val', 'records': len(saved),
        'violation_count': len(affected), 'violating_row_indices': affected.tolist(),
        'violating_record_ids': data['metadata']['record_id'][affected].tolist(),
        'maximum_absolute_difference': float(difference.max()), 'original_tolerance': {'atol': 1e-4, 'rtol': 1e-4},
        'historical_GPU_intermediates_available': False, 'details': []}
    if len(affected):
        # This high-precision reference is diagnostic arithmetic, not a new model or reported score.
        heads64 = LayerHeads().double().eval().requires_grad_(False)
        heads64.load_state_dict({name: value.double() for name, value in heads.state_dict().items()})
        with torch.inference_mode():
            logits64 = heads64(data['cls'][affected].double()).numpy()
        features64 = build_features(logits64, data['logits'][affected].numpy(), layers=12, k=5)
        x64 = normalize(features64, normalizer)
        with torch.inference_mode():
            score64heads = probe(torch.from_numpy(x64)).flatten().numpy()
        weights = probe.weight.detach().numpy().reshape(-1).astype(np.float64)
        bias = float(probe.bias.detach().numpy()[0])
        double_probe32features = x32[affected].astype(np.float64) @ weights + bias
        names = feature_names(12, 5)
        for local, index_in_role in enumerate(affected):
            final = data['logits'][index_in_role].numpy().astype(np.float64)
            sequence32 = np.concatenate((logits32[index_in_role].astype(np.float64), final[None]), axis=0)
            sequence64 = np.concatenate((logits64[local], final[None]), axis=0)
            ranks32, ranks64 = np.argsort(-sequence32, axis=1, kind='stable'), np.argsort(-sequence64, axis=1, kind='stable')
            sorted32 = np.take_along_axis(sequence32, ranks32, axis=1)
            predicted = int(final.argmax())
            competitor32, competitor64 = sequence32.copy(), sequence64.copy()
            competitor32[:, predicted] = -np.inf; competitor64[:, predicted] = -np.inf
            comp_ranks32, comp_ranks64 = np.argsort(-competitor32, axis=1, kind='stable'), np.argsort(-competitor64, axis=1, kind='stable')
            sorted_competitors = np.take_along_axis(competitor32, comp_ranks32, axis=1)
            feature_difference = features32[index_in_role].astype(np.float64) - features64[local].astype(np.float64)
            contributions = (x32[index_in_role].astype(np.float64) - x64[local].astype(np.float64)) * weights
            largest = np.argsort(-np.abs(contributions), kind='stable')[:10]
            result['diagnosis']['details'].append({
                'row_index': int(index_in_role), 'record_id': int(data['metadata']['record_id'][index_in_role]),
                'image_id': int(data['metadata']['image_id'][index_in_role]), 'saved_GPU_score': float(saved[index_in_role]),
                'CPU_FP32_score': float(cpu32[index_in_role]), 'CPU_FP64_heads_FP32_probe_score': float(score64heads[local]),
                'CPU_FP32_features_FP64_probe_score': float(double_probe32features[local]),
                'FP64_heads_matches_saved_at_original_tolerance': bool(np.isclose(score64heads[local], saved[index_in_role], atol=1e-4, rtol=1e-4)),
                'head_logits_FP32_vs_FP64_max_abs': float(np.max(np.abs(logits32[index_in_role] - logits64[local]))),
                'changed_top1_depths': (np.flatnonzero(ranks32[:, 0] != ranks64[:, 0]) + 1).tolist(),
                'changed_top5_set_depths': (np.flatnonzero(np.any(np.sort(ranks32[:, :5], axis=1) != np.sort(ranks64[:, :5], axis=1), axis=1)) + 1).tolist(),
                'changed_numeric_competitor_set_depths': (np.flatnonzero(np.any(np.sort(comp_ranks32[:, :5], axis=1) != np.sort(comp_ranks64[:, :5], axis=1), axis=1)) + 1).tolist(),
                'top1_minus_top2_margins': (sorted32[:, 0] - sorted32[:, 1]).tolist(),
                'top5_minus_top6_margins': (sorted32[:, 4] - sorted32[:, 5]).tolist(),
                'competitor5_minus_competitor6_margins': (sorted_competitors[:, 4] - sorted_competitors[:, 5]).tolist(),
                'numeric_feature_contribution_sum': float(contributions[:78].sum()),
                'dynamics_feature_contribution_sum': float(contributions[78:].sum()),
                'largest_feature_contribution_differences': [
                    {'feature': names[j], 'raw_FP32_minus_FP64': float(feature_difference[j]),
                     'weighted_normalized_difference': float(contributions[j]), 'probe_weight': float(weights[j]),
                     'normalizer_scale': float(normalizer['scale'][j])} for j in largest],
            })
    result['diagnosis']['interpretation_limit'] = ('FP32-versus-FP64 CPU comparisons characterize sensitivity. Historical GPU '
        'head logits/features were not saved, so exact historical GPU rank changes require a separately scoped original-backend replay.')
    result['complete'] = True
    result['original_CPU_replay_failure_cleared'] = False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--published-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    require(bool(os.environ.get('SLURM_JOB_ID')), 'Slurm allocation required')
    require(not os.environ.get('CUDA_VISIBLE_DEVICES'), 'CPU-only diagnostic requires empty CUDA_VISIBLE_DEVICES')
    require(not args.out.exists(), 'Existing diagnostic receipt must not be overwritten')
    signal.signal(signal.SIGALRM, deadline); signal.alarm(9 * 60)
    start = time.monotonic()
    result = {'complete': False, 'job_id': os.environ['SLURM_JOB_ID'], 'script_sha256': sha(__file__),
              'scope': 'Saved metric/schema checks and targeted seed17 validation CPU precision diagnostic; no fitting'}
    try:
        run(args, result)
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); result['elapsed_seconds'] = time.monotonic() - start
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(args.out.name + '.tmp.' + str(os.getpid()))
        with temporary.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, args.out)
        print(json.dumps({'complete': result['complete'], 'receipt': str(args.out)}), flush=True)


if __name__ == '__main__':
    main()
