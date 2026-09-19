"""Bounded real-CLS GPU fit/reload/resume smoke check; never a scientific fit."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from pilots.final_comparison_20260916.train import initialize
from .features import build_features, fit_normalizer, normalize
from .protocol import atomic_json, atomic_torch, campaign, read, require_slurm, sha256, verify
from .train import LayerHeads, _cpu_state, _rng_state, _restore_rng


def smoke(root):
    require_slurm()
    initialize(7)
    if not torch.cuda.is_available():
        raise RuntimeError("Preflight requires allocated CUDA")
    root = Path(root)
    campaign(root)
    directory = root / "preflight_cls"
    manifest = read(directory / "manifest.json")
    if not manifest.get("complete") or manifest["mode"] != "preflight":
        raise RuntimeError("Bounded parity extraction must finish before tiny-fit checks")
    payloads = []
    for spec in manifest["shards"]:
        verify(directory / spec["path"], spec["sha256"])
        payloads.append(torch.load(directory / spec["path"], map_location="cpu", weights_only=True))
    cls = torch.cat([p["cls"] for p in payloads]).float().cuda()
    labels = torch.cat([p["label"] for p in payloads]).cuda()
    final = torch.cat([p["logits"] for p in payloads]).numpy()
    heads = LayerHeads().cuda()
    optimizer = torch.optim.AdamW(heads.parameters(), lr=0.001, weight_decay=0)
    generator = torch.Generator().manual_seed(7)
    start = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    output = heads(cls)
    loss = sum(torch.nn.functional.cross_entropy(output[:, layer], labels) for layer in range(12))
    loss.backward()
    if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads.parameters()):
        raise FloatingPointError("Real-data auxiliary gradients failed")
    optimizer.step()
    snapshot = {"model": _cpu_state(heads), "optimizer": optimizer.state_dict(), "rng": _rng_state(generator)}
    atomic_torch(root / "tiny_resume.pt", snapshot)
    state = torch.load(root / "tiny_resume.pt", map_location="cpu", weights_only=True)
    restored = LayerHeads().cuda()
    restored.load_state_dict(state["model"])
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001, weight_decay=0)
    restored_optimizer.load_state_dict(state["optimizer"])
    _restore_rng(state["rng"], generator)
    with torch.no_grad():
        logits = heads(cls)
        torch.testing.assert_close(logits, restored(cls), atol=1e-6, rtol=1e-6)
    # The same next update after optimizer restoration must reproduce uninterrupted training.
    for model, optim in ((heads, optimizer), (restored, restored_optimizer)):
        optim.zero_grad(set_to_none=True)
        values = model(cls)
        sum(torch.nn.functional.cross_entropy(values[:, layer], labels) for layer in range(12)).backward()
        optim.step()
    with torch.no_grad():
        torch.testing.assert_close(heads(cls), restored(cls), atol=1e-6, rtol=1e-6)
    raw = build_features(logits.cpu().numpy(), final)
    normalizer = fit_normalizer(raw[:len(raw) // 2])
    x = torch.from_numpy(normalize(raw, normalizer)).cuda()
    targets = torch.cat([p["y"] for p in payloads]).float().cuda()
    if len(torch.unique(targets)) != 2:
        raise RuntimeError("Bounded real-data smoke sample lacks both error outcomes")
    probe = torch.nn.Linear(85, 1).cuda()
    optim = torch.optim.AdamW(probe.parameters(), lr=0.001, weight_decay=0.01)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(probe(x).flatten(), targets)
    loss.backward()
    optim.step()
    atomic_torch(root / "tiny_probe.pt", _cpu_state(probe))
    probe_restored = torch.nn.Linear(85, 1).cuda()
    probe_restored.load_state_dict(torch.load(root / "tiny_probe.pt", map_location="cpu", weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(probe(x), probe_restored(x), atol=1e-6, rtol=1e-6)
    torch.cuda.synchronize()
    seconds = time.monotonic() - start
    result = {"complete": True, "diagnostic_only": True, "records": len(cls), "features": raw.shape[1],
              "head_parameters": sum(p.numel() for p in heads.parameters()), "probe_parameters": sum(p.numel() for p in probe.parameters()),
              "two_head_updates_probe_update_reload_seconds": seconds, "resume_next_step_parity": True,
              "capture_seconds": manifest["elapsed_seconds"],
              "conservative_full_capture_seconds": manifest["elapsed_seconds"] * 28800 / len(cls),
              "projection_caveat": "Includes raw-data verification and large original-shard loading; startup is not amortized",
              "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "gpu": torch.cuda.get_device_name(),
              "campaign_sha256": sha256(root / "campaign.json"), "job_id": os.environ["SLURM_JOB_ID"]}
    atomic_json(root / "preflight_complete.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(smoke(args.root)), flush=True)


if __name__ == "__main__":
    main()
