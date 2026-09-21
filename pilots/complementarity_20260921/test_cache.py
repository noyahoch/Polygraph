"""Slurm-only semantic, serialization and exact next-update continuation checks."""
from __future__ import annotations
from pathlib import Path
import tempfile

from .common import COLUMNS, atomic_bytes, load_campaign, parent_modules, receipt, require, require_slurm
from .cache import expected_names, reference_features, semantic_indices


def run_tests(root, cuda=False):
    require_slurm()
    import numpy as np
    import torch
    from safetensors.torch import load_file, save
    from .ablation import cpu_state, save_resume, restore_resume
    campaign = load_campaign(root)
    _, features, _, original = parent_modules(campaign)
    names = expected_names()
    require(features.feature_names(12, 5) == names, 'Semantic names differ')
    require(names[66] == 'depth12_final_predicted_class_logit' and names[72] == 'depthclassifier_final_predicted_class_logit',
            'Column boundary differs')
    # Ties at the final-class decision and at the auxiliary top5 boundary.
    heads = np.zeros((3, 12, 100), dtype=np.float32)
    final = np.zeros((3, 100), dtype=np.float32)
    final[0, [7, 9]] = 4.0
    heads[0, :, 7] = -3.0
    final[1, 99] = 5.0
    heads[1, :, :6] = 2.0
    # Known competitor identities change along depth while chosen class is99.
    for depth in range(12):
        heads[1, depth, depth] = 3.0
    final[2, [0, 1]] = 1.0
    heads[2, :, [0, 1]] = 1.0
    actual = features.build_features(heads, final)
    reference = np.asarray([reference_features(heads[i], final[i]) for i in range(3)], dtype=np.float32)
    require(np.allclose(actual, reference, atol=1e-6, rtol=1e-6), 'Tied-logit semantic reference failed')
    require(actual[0, 0] == -3.0 and actual[0, 72] == 4.0,
            'Lower final-class-index tie winner must determine all depth chosen-class slots')
    require(actual[0, 73] == 4.0, 'Final competitor must exclude only chosenclass7 and retain tiedclass9')
    require(actual[2, 0] == 1.0 and actual[2, 1] == 1.0, 'Competitor selection excludes predictedclass0, retainsclass1')
    for arm, columns in COLUMNS.items():
        require(np.allclose(actual[:, columns], reference[:, columns], atol=1e-6, rtol=1e-6), 'A/B/C/D slice semantics failed: ' + arm)
    training = np.asarray([[1, 3, 7], [3, 3, 11]], dtype=np.float32)
    scaler = features.fit_normalizer(training)
    require(scaler['mean'] == [2, 3, 9] and scaler['scale'] == [1, 1, 2], 'Population/zero-scale normalization differs')
    normalized = features.normalize(training, scaler)
    require(np.array_equal(normalized, np.asarray([[-1, 0, -1], [1, 0, 1]], dtype=np.float32)), 'Train-only normalization fixture failed')
    metadata = {'record_id': np.asarray([15321, *range(9)]),
                'source_id': np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
                'severity': np.asarray([0, 0, 3, 5, 3, 5, 3, 5, 3, 5])}
    require(set(semantic_indices(metadata)) == set(range(10)), 'Semantic selection must retain9conditions+knownedgecase')
    details = {'feature_names': True, 'tied_logit_semantics': True, 'column_slices': True,
               'population_normalization_and_zero_scale': True, 'semantic_row_selection': True}
    if cuda:
        require(torch.cuda.is_available(), 'GPU preflight requires its own allocated CUDA device')
        original.initialize(7)
        device = torch.device('cuda')
        # Production checkpoint/restore functions are exercised with a small
        # linear objective; no scientific fitted model is modified.
        x = torch.arange(320, dtype=torch.float32).reshape(40, 8).div(320)
        y = (torch.arange(40) % 2).float()
        model = torch.nn.Linear(8, 1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        generator = torch.Generator().manual_seed(7)

        def update(network, optim, sampler):
            indices = torch.randperm(len(y), generator=sampler)[:24]
            optim.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(network(x[indices].to(device)).flatten(), y[indices].to(device))
            loss.backward()
            require(torch.isfinite(loss) and all(value.grad is not None and torch.isfinite(value.grad).all()
                                               for value in network.parameters()), 'Smoke loss/gradient invalid')
            optim.step()
            return indices

        update(model, optimizer, generator)
        before = cpu_state(model)
        directory = Path(root) / 'preflight'
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='resume-fixture-', dir=directory) as temporary:
            temporary = Path(temporary)
            identity = {'fixture': True, 'seed': 7, 'dimensions': 8}
            save_resume(temporary / 'resume.pt', model, optimizer, generator, original, identity=identity,
                        history=[{'epoch': 1}], best=before, best_ap=0.5, best_epoch=1, best_scores=torch.zeros(40))
            uninterrupted_order = update(model, optimizer, generator)
            uninterrupted = cpu_state(model)
            restored = torch.nn.Linear(8, 1).to(device)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001, weight_decay=0.01)
            restored_generator = torch.Generator().manual_seed(999)
            restore_resume(temporary / 'resume.pt', restored, restored_optimizer, restored_generator, original, identity)
            resumed_order = update(restored, restored_optimizer, restored_generator)
            require(torch.equal(uninterrupted_order, resumed_order), 'Sampler state restoration failed')
            require(all(torch.equal(value, cpu_state(restored)[name]) for name, value in uninterrupted.items()),
                    'Resumed next update differs from uninterrupted update')
            atomic_bytes(temporary / 'model.safetensors', save(cpu_state(restored)))
            reloaded = torch.nn.Linear(8, 1).to(device)
            reloaded.load_state_dict(load_file(str(temporary / 'model.safetensors'), device='cpu'))
            with torch.inference_mode():
                require(torch.equal(restored(x.to(device)), reloaded(x.to(device))), 'Portable save/reload failed')
            require(any(not torch.equal(value, uninterrupted[name]) for name, value in before.items()), 'Smoke update changed no parameters')
        details.update(next_optimizer_update_exact=True, sampler_restore_exact=True, portable_reload_exact=True)
    name = 'gpu_tests.json' if cuda else 'cpu_tests.json'
    return receipt(root, Path(root) / 'preflight' / name, [], checks=details, backend='cuda' if cuda else 'cpu')
