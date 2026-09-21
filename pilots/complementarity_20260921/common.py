"""Small immutable-manifest and parent-release adapters; no numerical imports."""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys

SCOPE = 'complementarity_20260921'
SEEDS = (7, 17, 27)
METADATA = ('record_id', 'image_id', 'source_id', 'severity', 'split_id', 'y', 'label', 'pred')
COLUMNS = {'A': list(range(72, 78)), 'B': list(range(66, 78)), 'C': list(range(78)), 'D': list(range(85))}
ROLES = ('probe_train', 'probe_val', 'dev_eval')
PARENT_SOURCE_SHA = 'd78438ba6a0c512ab66e5b59a16a772b0c4d972e39e9bf57ceb043b8936144b3'
PARENT_CAMPAIGN_SHA = 'f133671b92871129a0e859e47188c316e8ef535634a1fda33a9efa287dfa094f'
PARENT_SCORES_SHA = '8b17c6a55c26beb0dc9ac54c574849b61d93c8feedd576677f5e96c20f6dff75'
SPLIT_SALT = 'polygraph-complementarity-20260921:v1:'
RECIPE = {'epochs': 100, 'lr': 0.001, 'weight_decay': 0.01, 'batch_size': 256,
          'selection': 'strict greatest validation average_precision; earliest exact tie',
          'class_weight': 'probe_train negative/positive', 'dropout': 0,
          'normalization': 'probe_train per-column mean/population std, zero std replaced by1'}


def require_slurm():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('All preparation, numerical work and tests require a Slurm allocation')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, wanted):
    require(sha(path) == wanted, 'Checksum mismatch: ' + str(path))


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def frozen_json(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, 'Refusing to alter frozen input: ' + str(path))
    else:
        atomic_json(path, value)


def atomic_npz(path, **arrays):
    require_slurm()
    import numpy as np
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_bytes(path, stream.getvalue())


def atomic_torch(path, value):
    require_slurm()
    import torch
    stream = io.BytesIO()
    torch.save(value, stream)
    atomic_bytes(path, stream.getvalue())


@contextmanager
def lock(directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ('.' + name + '.lock')).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def source_identity():
    directory = Path(__file__).resolve().parent
    return {str(path): sha(path) for path in sorted(directory.glob('*.py'))}


def prepare(root, parent_ld_root, parent_ld_release, split_salt, protocol_path):
    require_slurm()
    root, ld, release = (Path(path).resolve() for path in (root, parent_ld_root, parent_ld_release))
    require(root != ld and not root.is_relative_to(ld), 'New experiment must have an independent root')
    verify(ld / 'campaign.json', PARENT_CAMPAIGN_SHA)
    verify(ld / 'evaluation/scores.npz', PARENT_SCORES_SHA)
    verify(release / 'source_manifest.json', PARENT_SOURCE_SHA)
    parent = read(ld / 'campaign.json')
    baseline = Path(parent['baseline_root'])
    ld_roles, old_roles = read(ld / 'role_map.json')['roles'], read(baseline / 'role_map.json')['roles']
    old_base = set(old_roles['base_train']['photo_ids'])
    old_meta = set(old_roles['meta']['photo_ids'])
    require(set(ld_roles['head_train']['photo_ids']) <= old_base, 'Unexpected auxiliary-head sources')
    require(old_meta <= set(ld_roles['probe_train']['photo_ids']), 'Expected old-meta inclusion changed')
    require(set(ld_roles['head_train']['photo_ids']) | set(ld_roles['probe_train']['photo_ids']) == old_base | old_meta,
            'LD fitting-source union differs from recorded allocation')
    require(ld_roles['probe_val']['photo_ids'] == sorted(old_roles['checkpoint']['photo_ids']), 'Checkpoint roles differ')
    photos = ld_roles['dev_eval']['photo_ids']
    require(set(photos) == set(old_roles['dev_eval']['photo_ids']) and len(photos) == 800, 'Development cohort changed')
    require(split_salt == SPLIT_SALT, 'Approved exact fusion split namespace is required')
    ordered = sorted(photos, key=lambda photo: (hashlib.sha256((split_salt + str(photo)).encode()).hexdigest(), photo))
    hashes = {}
    for directory, names in ((ld, ('campaign.json', 'role_map.json', 'evaluation_gate.json', 'cls/manifest.json',
                                    'evaluation/scores.npz', 'evaluation/complete.json')),
                             (baseline, ('campaign.json', 'role_map.json', 'evaluation/scores.npz', 'evaluation/complete.json')),
                             (release, ('source_manifest.json',))):
        for name in names:
            hashes[str(directory / name)] = sha(directory / name)
    for seed in SEEDS:
        for relative in ('heads/model.safetensors', 'heads/checkpoint.pt', 'heads/config.json', 'heads/complete.json',
                         'probe/model.safetensors', 'probe/checkpoint.pt', 'probe/normalizer.json', 'probe/config.json',
                         'probe/complete.json', 'probe/validation.npz', 'predictions/dev_eval.npz'):
            path = ld / 'runs' / f'seed{seed}' / relative
            hashes[str(path)] = sha(path)
    value = {
        'scope_id': SCOPE, 'root': str(root), 'seeds': list(SEEDS), 'metadata_keys': list(METADATA),
        'parent_ld_root': str(ld), 'parent_ld_release': str(release), 'parent_baseline_root': str(baseline),
        'parent_hashes': hashes, 'source_files': source_identity(),
        'fusion_split_namespace': split_salt, 'fusion_fit_photo_ids': sorted(ordered[:400]),
        'fusion_assessment_photo_ids': sorted(ordered[400:]),
        'fusion_seed_partner': {'7': 17, '17': 27, '27': 7},
        'ablation_columns': COLUMNS, 'ablation_recipe': RECIPE,
        'roles': {role: ld_roles[role] for role in ROLES},
        'semantic_rows': 'lowest record_id for each of nine conditions in each role plus15321 when present, every seed',
        'semantic_conditions': [[0, 0]] + [[source, severity] for source in range(1, 5) for severity in (3, 5)],
        'protocol_sha256': sha(protocol_path), 'protocol_filename': Path(protocol_path).name,
        'scientific_output_policy': 'new namespace only; fixed original D and all original models preserved',
        'original_test_access': False,
    }
    with lock(root, 'prepare'):
        frozen_json(root / 'campaign.json', value)
    return value


def load_campaign(root):
    require_slurm()
    root = Path(root).resolve()
    value = read(root / 'campaign.json')
    require(value['scope_id'] == SCOPE and value['root'] == str(root) and value['seeds'] == list(SEEDS)
            and value['ablation_columns'] == COLUMNS and value['ablation_recipe'] == RECIPE,
            'Frozen follow-up contract changed')
    require(value['source_files'] == source_identity(), 'Follow-up source release changed')
    for path, wanted in value['parent_hashes'].items():
        verify(path, wanted)
    return value


def parent_modules(campaign):
    require_slurm()
    release = Path(campaign['parent_ld_release'])
    verify(release / 'source_manifest.json', PARENT_SOURCE_SHA)
    for name, wanted in read(release / 'source_manifest.json')['files'].items():
        verify(release / name, wanted)
    if str(release) not in sys.path:
        sys.path.insert(0, str(release))
    from pilots.logit_dynamics_20260919 import data, features, protocol, train
    require(Path(protocol.__file__).resolve().is_relative_to(release), 'LD code was imported outside its frozen release')
    protocol.campaign(campaign['parent_ld_root'])
    protocol.evaluation_gate(campaign['parent_ld_root'])
    return data, features, protocol, train


def receipt(root, path, files, **details):
    root = Path(root).resolve()
    entries = {}
    for file in files:
        file = Path(file).resolve()
        require(file.is_relative_to(root), 'New receipt may bind only files in its own namespace')
        entries[str(file.relative_to(root))] = sha(file)
    value = {'complete': True, 'campaign_sha256': sha(root / 'campaign.json'), 'files': entries, **details}
    frozen_json(path, value)
    return value


def verify_receipt(root, path):
    root, path = Path(root).resolve(), Path(path)
    if not path.is_absolute():
        path = root / path
    value = read(path)
    require(value['complete'] is True and value['campaign_sha256'] == sha(root / 'campaign.json'), 'Invalid completion binding')
    for relative, wanted in value['files'].items():
        file = (root / relative).resolve()
        require(file.is_relative_to(root), 'Receipt path escapes its namespace')
        verify(file, wanted)
    return value


def freeze_all(root):
    load_campaign(root)
    root = Path(root).resolve()
    files = []
    for name in ('cache/complete.json', 'ablation/freeze.json', 'fusion/freeze.json'):
        value = verify_receipt(root, name)
        files.append(root / name)
        files.extend(root / relative for relative in value['files'])
    return receipt(root, root / 'evaluation_gate.json', files,
                   all_models_frozen_before_assessment=True, original_models_unchanged=True)
