"""Allocated-GPU gates: canonical raw parity, feature semantics, and exact resume."""
from __future__ import annotations
import argparse
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
import unittest
import numpy as np
import torch
from .extract import extract
from .features import build_features, feature_names, fit_normalizer, normalize
from .protocol import (atomic_json, atomic_torch, campaign, initialize, lock, read, require_slurm, sha256)
from .train import LayerHeads, _cpu_state, _resume, _rng_state
from .evaluate import weighted_fixture_checks


def numerical_checks():
    require_slurm()
    initialize(7)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU preflight is required")
    # Reuse hand-calculated feature expectations independently of new data plumbing.
    from pilots.logit_dynamics_20260919.test_science import PaperFeatureCases, NormalizationCases
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(PaperFeatureCases),
                               unittest.defaultTestLoader.loadTestsFromTestCase(NormalizationCases)])
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Preserved scientific feature tests failed:\n" + output.getvalue())
    names = feature_names()
    if (len(names) != 85 or names[66] != "depth12_final_predicted_class_logit"
            or names[72] != "depthclassifier_final_predicted_class_logit"
            or names[78] != "top1_switch_rate"):
        raise RuntimeError("Semantic feature order changed")
    heads = LayerHeads().cuda()
    if sum(p.numel() for p in heads.parameters()) != 922800:
        raise RuntimeError("Auxiliary head architecture changed")
    cls = torch.randn(8, 12, 768, device="cuda")
    labels = torch.arange(8, device="cuda")
    optimizer = torch.optim.AdamW(heads.parameters(), lr=0.001, weight_decay=0.0)
    prior = [head.weight.detach().clone() for head in heads.heads]
    outputs = heads(cls)
    loss = sum(torch.nn.functional.cross_entropy(outputs[:, i], labels) for i in range(12))
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in heads.parameters()):
        raise RuntimeError("Head gradients are invalid")
    optimizer.step()
    if any(torch.equal(before, head.weight) for before, head in zip(prior, heads.heads)):
        raise RuntimeError("A head failed to update")
    with torch.no_grad():
        outputs = heads(cls).cpu().numpy()
    final = np.zeros((8, 100), dtype=np.float32)
    final[np.arange(8), np.arange(8)] = 1
    features = build_features(outputs, final)
    scaler = fit_normalizer(features[:6])
    x = torch.from_numpy(normalize(features, scaler)).cuda()
    probe = torch.nn.Linear(85, 1).cuda()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.001, weight_decay=0.01)
    generator = torch.Generator().manual_seed(7)
    targets = torch.tensor([0, 1] * 4, device="cuda", dtype=torch.float32)

    def step(model, optim, gen):
        order = torch.randperm(8, generator=gen).cuda()
        optim.zero_grad(set_to_none=True)
        value = torch.nn.functional.binary_cross_entropy_with_logits(model(x[order]).flatten(), targets[order])
        value.backward()
        if not torch.isfinite(value) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Probe gradient/finite gate failed")
        optim.step()

    step(probe, optimizer, generator)
    identity = {"synthetic_resume_gate": 1}
    state = {"identity": identity, "model": _cpu_state(probe), "optimizer": copy.deepcopy(optimizer.state_dict()),
             "rng": _rng_state(generator)}
    with tempfile.TemporaryDirectory(prefix="canonical-ld-preflight-") as tmp:
        atomic_torch(Path(tmp) / "resume.pt", state)
        step(probe, optimizer, generator)
        expected = _cpu_state(probe)
        restored = torch.nn.Linear(85, 1).cuda()
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001, weight_decay=0.01)
        restored_generator = torch.Generator().manual_seed(999)
        _resume(Path(tmp), identity, restored, restored_optimizer, restored_generator)
        step(restored, restored_optimizer, restored_generator)
        for key, wanted in expected.items():
            torch.testing.assert_close(restored.state_dict()[key].cpu(), wanted, atol=0, rtol=0)
        with torch.no_grad():
            torch.testing.assert_close(restored(x), probe(x), atol=1e-6, rtol=1e-6)
        try:
            _resume(Path(tmp), {"different": True}, restored, restored_optimizer, restored_generator)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Resume accepted a changed identity")
    return {"scientific_feature_checks": result.testsRun, "semantic_order": True,
            "finite_gradients": True, "save_load": True, "resume_exact": True,
            "resume_identity_rejected": True, "test_output": output.getvalue(), **weighted_fixture_checks()}


def preflight(root, data_root):
    require_slurm()
    root = Path(root).resolve()
    campaign(root)
    directory = root / "preflight"
    with lock(directory, "preflight"):
        if (directory / "complete.json").exists():
            value = read(directory / "complete.json")
            if value.get("complete") is True and value["campaign_sha256"] == sha256(root / "campaign.json"):
                return value
            raise RuntimeError("Pre-existing preflight receipt identity differs")
        attempt = f"{os.environ['SLURM_JOB_ID']}_{time.time_ns()}"
        began = time.monotonic()
        try:
            captured = extract(root, data_root, "preflight")
            checks = numerical_checks()
            value = {"complete": True, "campaign_sha256": sha256(root / "campaign.json"),
                     "capture_manifest_sha256": sha256(root / "preflight_cls" / "manifest.json"),
                     "capture_records": captured["records"], "parity": captured["parity_maximum"],
                     "checks": checks, "job_id": os.environ["SLURM_JOB_ID"],
                     "elapsed_seconds": time.monotonic() - began}
            atomic_json(directory / f"attempt_{attempt}.json", value)
            atomic_json(directory / "complete.json", value)
            return value
        except Exception as exc:
            atomic_json(directory / f"attempt_{attempt}.json", {"complete": False,
                "campaign_sha256": sha256(root / "campaign.json"), "exception": repr(exc),
                "traceback": traceback.format_exc(), "job_id": os.environ["SLURM_JOB_ID"]})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(args.root, args.data_root)), flush=True)


if __name__ == "__main__":
    main()
