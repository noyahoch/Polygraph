"""Nine fixed linear probes; original auxiliary heads and full D stay frozen."""
from __future__ import annotations
import os
from pathlib import Path
import signal
import time

from .common import (COLUMNS, METADATA, RECIPE, SEEDS, atomic_bytes, atomic_json, atomic_npz, atomic_torch,
                     frozen_json, load_campaign, lock, parent_modules, read, receipt, require,
                     require_slurm, sha, verify_receipt)
from .cache import expected_names, load_cached

STOP = False


def _stop(_signal, _frame):
    global STOP
    STOP = True


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_resume(path, model, optimizer, generator, parent_train, *, identity, history, best, best_ap, best_epoch, best_scores):
    atomic_torch(path, {'identity': identity, 'model': cpu_state(model), 'optimizer': optimizer.state_dict(),
                       'rng': parent_train._rng_state(generator), 'history': history, 'best': best,
                       'best_ap': best_ap, 'best_epoch': best_epoch, 'best_scores': best_scores})


def restore_resume(path, model, optimizer, generator, parent_train, identity):
    require_slurm()
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    require(state['identity'] == identity, 'Resume identity changed')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    parent_train._restore_rng(state['rng'], generator)
    require([row['epoch'] for row in state['history']] == list(range(1, len(state['history']) + 1)), 'Resume history has holes')
    return state


def fit(root, seed):
    require_slurm()
    import numpy as np
    import torch
    from safetensors.torch import load_file, save
    from sklearn.metrics import average_precision_score
    root = Path(root).resolve()
    require(seed in SEEDS, 'Unapproved seed')
    campaign = load_campaign(root)
    verify_receipt(root, 'cache/complete.json')
    verify_receipt(root, 'statistics/draw_manifest.json')
    require(not (root / 'evaluation_gate.json').exists(), 'No new probe fitting after the global freeze')
    require(torch.cuda.is_available(), 'All new probes use the original FP32 CUDA training path')
    _, features, _, original = parent_modules(campaign)
    parent = Path(campaign['parent_ld_root'])
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    train_data, val_data = (load_cached(root, seed, role) for role in ('probe_train', 'probe_val'))
    labels = train_data['y']
    positive, negative = int(labels.sum()), len(labels) - int(labels.sum())
    require(positive > 0 and negative > 0 and len(np.unique(val_data['y'])) == 2, 'Both error outcomes are required')
    results = []
    for arm in ('A', 'B', 'C'):
        directory = root / 'runs' / 'ablation' / arm / f'seed{seed}'
        with lock(directory, 'fit'):
            if (directory / 'complete.json').exists():
                results.append(verify_receipt(root, directory / 'complete.json'))
                continue
            columns = COLUMNS[arm]
            normalizer = features.fit_normalizer(train_data['raw_features'][:, columns])
            xtrain = torch.from_numpy(features.normalize(train_data['raw_features'][:, columns], normalizer))
            xval = torch.from_numpy(features.normalize(val_data['raw_features'][:, columns], normalizer))
            targets = torch.from_numpy(labels.astype(np.float32))
            frozen_json(directory / 'normalizer.json', normalizer)
            identity = {'campaign_sha256': sha(root / 'campaign.json'), 'cache_gate_sha256': sha(root / 'cache/complete.json'),
                        'seed': seed, 'arm': arm, 'columns': columns, 'recipe': RECIPE,
                        'normalizer_sha256': sha(directory / 'normalizer.json'),
                        'features': [expected_names()[column] for column in columns],
                        'training_role': 'probe_train', 'selection_role': 'probe_val',
                        'training_records': len(labels), 'positive_weight': negative / positive,
                        'initialization': 'original initialize(seed), frozen load_heads constructor RNG prefix, then readout',
                        'head_checkpoint_sha256': sha(parent / 'runs' / f'seed{seed}' / 'heads/checkpoint.pt')}
            frozen_json(directory / 'config.json', identity)
            original.initialize(seed)
            # Preserve the original fit_probe RNG prefix even though feature
            # matrices have been cached. These heads are never updated.
            frozen_heads = original.load_heads(parent, seed, torch.device('cuda'))
            del frozen_heads
            model = torch.nn.Linear(len(columns), 1).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01,
                                           betas=(0.9, 0.999), eps=1e-8)
            generator = torch.Generator().manual_seed(seed)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device='cuda'))
            history, best, best_ap, best_epoch, best_scores = [], None, -1.0, None, None
            if (directory / 'resume.pt').exists():
                state = restore_resume(directory / 'resume.pt', model, optimizer, generator, original, identity)
                history, best, best_ap, best_epoch, best_scores = (state[key] for key in ('history', 'best', 'best_ap', 'best_epoch', 'best_scores'))
            began = time.monotonic()
            for epoch in range(len(history) + 1, 101):
                model.train()
                order = torch.randperm(len(targets), generator=generator)
                loss_sum = 0.0
                for start in range(0, len(order), 256):
                    indices = order[start:start + 256]
                    optimizer.zero_grad(set_to_none=True)
                    scores = model(xtrain[indices].cuda()).flatten()
                    loss = loss_fn(scores, targets[indices].cuda())
                    loss.backward()
                    require(torch.isfinite(loss) and all(param.grad is not None and torch.isfinite(param.grad).all()
                                                         for param in model.parameters()), 'Invalid loss/gradients')
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(indices)
                scores = original.probe_scores(model, xval, torch.device('cuda'), batch_size=512)
                ap = float(average_precision_score(val_data['y'], scores))
                if ap > best_ap:
                    best_ap, best_epoch, best = ap, epoch, cpu_state(model)
                    best_scores = torch.from_numpy(scores.copy())
                history.append({'epoch': epoch, 'training_loss': loss_sum / len(targets),
                                'validation_average_precision': ap, 'best_epoch': best_epoch,
                                'best_average_precision': best_ap})
                save_resume(directory / 'resume.pt', model, optimizer, generator, original, identity=identity,
                            history=history, best=best, best_ap=best_ap, best_epoch=best_epoch, best_scores=best_scores)
                atomic_json(directory / 'history.json', history)
                if epoch % 10 == 0:
                    # Progress only; new assessment effects are forbidden here.
                    print(f'probe training: {arm}/seed{seed} epoch{epoch}/100', flush=True)
                if STOP:
                    raise InterruptedError('Committed epoch saved; resume the identical seed job')
            require(len(history) == 100 and best is not None, 'Incomplete fixed-budget probe')
            atomic_torch(directory / 'checkpoint.pt', {'identity': identity, 'state_dict': best, 'epoch': best_epoch})
            atomic_bytes(directory / 'model.safetensors', save({key: value.contiguous() for key, value in best.items()}))
            reloaded = torch.nn.Linear(len(columns), 1).cuda().eval().requires_grad_(False)
            reloaded.load_state_dict(load_file(str(directory / 'model.safetensors'), device='cpu'))
            restored = original.probe_scores(reloaded, xval, torch.device('cuda'), batch_size=512)
            require(np.allclose(restored, best_scores.numpy(), atol=1e-6, rtol=1e-6), 'Selected portable model reload differs')
            atomic_npz(directory / 'validation.npz', **{key: val_data[key] for key in METADATA}, score=restored)
            files = [directory / name for name in ('config.json', 'normalizer.json', 'history.json', 'resume.pt',
                                                   'checkpoint.pt', 'model.safetensors', 'validation.npz')]
            results.append(receipt(root, directory / 'complete.json', files, arm=arm, seed=seed, epochs=100,
                                   selected_epoch=best_epoch, selected_validation_ap=best_ap,
                                   portable_reload_passed=True, elapsed_current_attempt_seconds=time.monotonic() - began,
                                   job_id=os.environ['SLURM_JOB_ID']))
    return results


def freeze(root):
    load_campaign(root)
    root = Path(root).resolve()
    files = []
    for arm in ('A', 'B', 'C'):
        for seed in SEEDS:
            path = root / 'runs' / 'ablation' / arm / f'seed{seed}' / 'complete.json'
            value = verify_receipt(root, path)
            require(value['epochs'] == 100 and value['arm'] == arm and value['seed'] == seed
                    and value['portable_reload_passed'] is True, 'Incomplete ablation fit')
            files.append(path)
            files.extend(root / name for name in value['files'])
    return receipt(root, root / 'ablation/freeze.json', files, fits=9, original_D_reused=True)


def predict(root):
    require_slurm()
    import numpy as np
    import torch
    from safetensors.torch import load_file
    root = Path(root).resolve()
    campaign = load_campaign(root)
    verify_receipt(root, 'evaluation_gate.json')
    require(torch.cuda.is_available(), 'Final probe predictions use original CUDA arithmetic')
    complete_path = root / 'predictions/ablation_complete.json'
    with lock(root / 'predictions', 'ablation'):
        if complete_path.exists():
            return verify_receipt(root, complete_path)
        _, features, _, original = parent_modules(campaign)
        parent = Path(campaign['parent_ld_root'])
        metadata, all_scores = None, {}
        for seed in SEEDS:
            original.initialize(seed)
            data = load_cached(root, seed, 'dev_eval')
            if metadata is None:
                metadata = {key: data[key] for key in METADATA}
            require(all(np.array_equal(metadata[key], data[key]) for key in METADATA), 'Seed metadata differ')
            for arm in ('A', 'B', 'C'):
                directory = root / 'runs' / 'ablation' / arm / f'seed{seed}'
                scaler = read(directory / 'normalizer.json')
                x = torch.from_numpy(features.normalize(data['raw_features'][:, COLUMNS[arm]], scaler))
                model = torch.nn.Linear(len(COLUMNS[arm]), 1).cuda().eval().requires_grad_(False)
                model.load_state_dict(load_file(str(directory / 'model.safetensors'), device='cpu'))
                all_scores[f'{arm}/seed{seed}'] = original.probe_scores(model, x, torch.device('cuda'), batch_size=512)
            with np.load(parent / 'runs' / f'seed{seed}' / 'predictions/dev_eval.npz', allow_pickle=False) as archive:
                require(all(np.array_equal(metadata[key], archive[key]) for key in METADATA), 'Original D metadata differ')
                all_scores[f'D/seed{seed}'] = archive['score'].copy()
        keys = [f'{arm}/seed{seed}' for arm in ('A', 'B', 'C', 'D') for seed in SEEDS]
        path = root / 'predictions/ablation.npz'
        require(not path.exists(), 'Uncommitted prediction output is preserved; explicit technical recovery required')
        atomic_npz(path, **metadata, score_keys=np.asarray(keys), scores=np.stack([all_scores[key] for key in keys], axis=1))
        return receipt(root, complete_path, [path], records=7200, source_photographs=800, models=12,
                       assessment_metrics_not_computed=True, evaluation_gate_sha256=sha(root / 'evaluation_gate.json'))
