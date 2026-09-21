"""Frozen-head 85-feature cache with independent semantic and D-replay gates."""
from __future__ import annotations
from collections import Counter
import math
from pathlib import Path
import time

from .common import (COLUMNS, METADATA, ROLES, SEEDS, atomic_json, atomic_npz, load_campaign,
                     lock, parent_modules, read, receipt, require, require_slurm, sha, verify, verify_receipt)

DYNAMICS = ('top1_switch_rate', 'topk_weighted_jaccard', 'unique_topk_count',
            'top1_mode_frequency', 'top1_entropy', 'top1_unique_count', 'top1_commitment_depth')


def expected_names():
    names = []
    for depth in [*range(1, 13), 'classifier']:
        names.append(f'depth{depth}_final_predicted_class_logit')
        names.extend(f'depth{depth}_competitor_rank{rank}_logit' for rank in range(1, 6))
    return names + list(DYNAMICS)


def reference_features(head_values, final_values):
    """Independent scalar/list formulation; stable ties explicitly use class ID."""
    require_slurm()
    sequence = [list(map(float, values)) for values in head_values] + [list(map(float, final_values))]
    require(len(sequence) == 13 and all(len(row) == 100 for row in sequence), 'Reference shape mismatch')
    require(all(math.isfinite(value) for row in sequence for value in row), 'Nonfinite semantic reference')
    predicted = max(range(100), key=lambda cls: (sequence[-1][cls], -cls))
    numeric, top1, top_sets, distributions = [], [], [], []
    for values in sequence:
        ranks = sorted(range(100), key=lambda cls: (-values[cls], cls))
        competitors = [cls for cls in ranks if cls != predicted][:5]
        numeric.extend([values[predicted], *(values[cls] for cls in competitors)])
        top1.append(ranks[0])
        top_sets.append(set(ranks[:5]))
        peak = values[ranks[0]]
        weights = {cls: math.exp(values[cls] - peak) for cls in ranks[:5]}
        total = sum(weights.values())
        distributions.append({cls: value / total for cls, value in weights.items()})
    switches = sum(left != right for left, right in zip(top1[:-1], top1[1:])) / 12
    overlap = []
    for left, right in zip(distributions[:-1], distributions[1:]):
        shared = sum(min(left.get(cls, 0), right.get(cls, 0)) for cls in left.keys() | right.keys())
        overlap.append(shared / (2 - shared))
    frequencies = Counter(top1)
    last_mismatch = max((depth + 1 for depth, cls in enumerate(top1) if cls != predicted), default=0)
    dynamics = [switches, sum(overlap) / 12, len(set().union(*top_sets)), max(frequencies.values()) / 13,
                -sum((count / 13) * math.log(count / 13) for count in frequencies.values()),
                len(frequencies), last_mismatch / 12]
    return numeric + dynamics


def semantic_indices(metadata):
    require_slurm()
    wanted = [(0, 0)] + [(source, severity) for source in range(1, 5) for severity in (3, 5)]
    result = []
    for condition in wanted:
        candidates = [i for i, pair in enumerate(zip(metadata['source_id'], metadata['severity'])) if tuple(pair) == condition]
        require(candidates, 'A frozen input condition is absent from semantic fixtures')
        result.append(min(candidates, key=lambda i: int(metadata['record_id'][i])))
    result.extend(i for i, value in enumerate(metadata['record_id']) if value == 15321)
    return sorted(set(result))


def check_metadata(metadata, spec):
    require_slurm()
    import numpy as np
    require(metadata['record_id'].tolist() == spec['record_ids'], 'Role records differ')
    require(all(metadata[key].dtype == np.int64 for key in METADATA), 'Metadata dtype differs')
    photos, counts = np.unique(metadata['image_id'], return_counts=True)
    require(photos.tolist() == sorted(spec['photo_ids']) and np.all(counts == 9), 'Nine-view source groups differ')
    require(np.array_equal(metadata['y'], (metadata['pred'] != metadata['label']).astype(np.int64)), 'Error label changed')


def load_cached(root, seed, role, *, gated=True):
    require_slurm()
    import numpy as np
    root = Path(root)
    if gated:
        verify_receipt(root, 'cache/complete.json')
    path = root / 'cache' / f'seed{seed}' / f'{role}.npz'
    item = read(path.with_suffix('.json'))
    verify(path, item['sha256'])
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    require(values['feature_names'].tolist() == expected_names(), 'Semantic column names/order changed')
    require(values['raw_features'].dtype == np.float32 and values['raw_features'].shape == (len(values['y']), 85)
            and np.isfinite(values['raw_features']).all(), 'Invalid feature cache')
    return values


def build(root):
    require_slurm()
    import numpy as np
    import torch
    from safetensors.torch import load_file
    root = Path(root).resolve()
    campaign = load_campaign(root)
    verify_receipt(root, 'statistics/draw_manifest.json')
    directory = root / 'cache'
    with lock(directory, 'build'):
        if (directory / 'complete.json').exists():
            return verify_receipt(root, directory / 'complete.json')
        data_module, features, _, train = parent_modules(campaign)
        require(torch.cuda.is_available(), 'Frozen CUDA feature path is required')
        require(features.feature_names(12, 5) == expected_names(), 'Parent feature semantics changed')
        require(expected_names()[78:] == list(DYNAMICS), 'Dynamics names changed')
        parent = Path(campaign['parent_ld_root'])
        all_files, gate_rows = [], []
        started = time.monotonic()
        for seed in SEEDS:
            train.initialize(seed)
            heads = train.load_heads(parent, seed, torch.device('cuda'))
            probe = torch.nn.Linear(85, 1).cuda().eval().requires_grad_(False)
            probe.load_state_dict(load_file(str(parent / 'runs' / f'seed{seed}' / 'probe/model.safetensors'), device='cpu'))
            original_scaler = read(parent / 'runs' / f'seed{seed}' / 'probe/normalizer.json')
            for role in ROLES:
                path = directory / f'seed{seed}' / f'{role}.npz'
                sidecar = path.with_suffix('.json')
                role_gate = path.with_suffix('.gate.json')
                if role_gate.exists():
                    # A fully checked prior role can be resumed; no partial file is
                    # silently treated as a complete cache compatibility check.
                    prior = verify_receipt(root, role_gate)
                    gate_rows.append(prior['check'])
                    all_files.extend([path, sidecar, role_gate, path.with_suffix('.semantic.npz'), path.with_suffix('.check.json')])
                    continue
                require(not path.exists() and not sidecar.exists(), 'Partial cache role preserved; use a documented technical recovery')
                data = data_module.load_role(parent, role)
                check_metadata(data['metadata'], campaign['roles'][role])
                chosen = semantic_indices(data['metadata'])
                batches = []
                with torch.inference_mode():
                    for start in range(0, len(data['cls']), 512):
                        batches.append(heads(data['cls'][start:start + 512].float().cuda()).cpu().numpy())
                head_values = np.concatenate(batches)
                require(np.isfinite(head_values).all(), 'Nonfinite auxiliary-head logits')
                raw = features.build_features(head_values, data['logits'].numpy(), layers=12, k=5)
                independent = np.asarray([reference_features(head_values[i], data['logits'][i].tolist()) for i in chosen], dtype=np.float32)
                require(np.allclose(raw[chosen], independent, atol=1e-6, rtol=1e-6), 'Independent feature semantics disagree')
                atomic_npz(path, **data['metadata'], raw_features=raw, feature_names=np.asarray(expected_names()))
                atomic_json(sidecar, {'sha256': sha(path), 'seed': seed, 'role': role,
                    'campaign_sha256': sha(root / 'campaign.json'), 'parent_manifest_sha256': sha(parent / 'cls/manifest.json'),
                    'head_checkpoint_sha256': sha(parent / 'runs' / f'seed{seed}' / 'heads/checkpoint.pt'),
                    'feature_names': expected_names(), 'records': len(raw)})
                evidence = path.with_suffix('.semantic.npz')
                atomic_npz(evidence, record_id=data['metadata']['record_id'][chosen],
                           source_id=data['metadata']['source_id'][chosen], severity=data['metadata']['severity'][chosen],
                           auxiliary_logits=head_values[chosen], classifier_logits=data['logits'][chosen].numpy(),
                           independently_derived_features=independent, cached_features=raw[chosen])
                # Compatibility is checked through a new disk reload, not the RAM
                # matrix used by the producing call.
                cached = load_cached(root, seed, role, gated=False)
                check_metadata(cached, campaign['roles'][role])
                check = {'seed': seed, 'role': role, 'records': len(raw), 'feature_names_exact': True,
                         'semantic_record_ids': cached['record_id'][chosen].tolist(),
                         'semantic_maximum_absolute_difference': float(np.max(np.abs(raw[chosen] - independent))),
                         'cache_sha256': sha(path), 'column_slices': COLUMNS}
                if role == 'probe_train':
                    scaler = features.fit_normalizer(cached['raw_features'])
                    check['normalizer'] = {}
                    for name in ('mean', 'scale'):
                        delta = np.abs(np.asarray(scaler[name]) - original_scaler[name])
                        passed = bool(np.allclose(scaler[name], original_scaler[name], atol=1e-4, rtol=1e-4))
                        check['normalizer'][name] = {'passed': passed, 'maximum_absolute_difference': float(delta.max())}
                    atomic_json(path.with_suffix('.check.json'), check)
                    require(all(item['passed'] for item in check['normalizer'].values()), 'Original training normalizer mismatch')
                else:
                    values = torch.from_numpy(features.normalize(cached['raw_features'], original_scaler))
                    scores = train.probe_scores(probe, values, torch.device('cuda'), batch_size=512)
                    reference_path = parent / 'runs' / f'seed{seed}' / ('probe/validation.npz' if role == 'probe_val' else 'predictions/dev_eval.npz')
                    with np.load(reference_path, allow_pickle=False) as archive:
                        for name in METADATA:
                            require(np.array_equal(cached[name], archive[name]), 'Parent/new-cache metadata mismatch')
                        expected = archive['score']
                        violations = np.flatnonzero(~np.isclose(scores, expected, atol=1e-4, rtol=1e-4))
                        check['D_replay'] = {'passed': len(violations) == 0, 'violations': cached['record_id'][violations].tolist(),
                                             'maximum_absolute_difference': float(np.max(np.abs(scores - expected))),
                                             'atol': 1e-4, 'rtol': 1e-4, 'reference_sha256': sha(reference_path)}
                    atomic_json(path.with_suffix('.check.json'), check)
                    require(check['D_replay']['passed'], 'D replay FROM NEW CACHE exceeds unchanged tolerance')
                receipt(root, role_gate, [path, sidecar, evidence, path.with_suffix('.check.json')], check=check)
                gate_rows.append(check)
                all_files.extend([path, sidecar, role_gate, evidence, path.with_suffix('.check.json')])
                print(f'cache compatibility passed: seed{seed}/{role}', flush=True)
                del data, raw, cached, head_values, batches
        return receipt(root, directory / 'complete.json', all_files, checks=gate_rows,
                       new_cache_D_replay_all_three_validation_and_development=True,
                       elapsed_seconds=time.monotonic() - started)
