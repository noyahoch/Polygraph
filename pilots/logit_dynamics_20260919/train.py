"""Fixed auxiliary heads, then a train-normalized linear error probe; resumable per epoch."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from pilots.final_comparison_20260916.train import initialize

from .data import load_role
from .features import build_features, feature_names, fit_normalizer, normalize
from .protocol import (METADATA, RECIPE, atomic_bytes, atomic_json, atomic_npz, atomic_torch, campaign,
                       evaluation_gate, frozen_json, lock, read, require_slurm, run_dir, sha256, verify)

STOP = False


class LayerHeads(torch.nn.Module):
    """Twelve independent class-supervised linear maps; no shared projection or logit lens."""
    def __init__(self):
        super().__init__()
        self.heads = torch.nn.ModuleList(torch.nn.Linear(768, 100) for _ in range(12))

    def forward(self, cls):
        return torch.stack([head(cls[:, layer]) for layer, head in enumerate(self.heads)], dim=1)


def _device():
    require_slurm()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires allocated CUDA; no local/CPU fallback")
    return torch.device("cuda")


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _portable(path, state):
    from safetensors.torch import save
    atomic_bytes(path, save({key: value.contiguous() for key, value in state.items()}))


def _rng_state(generator):
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "loader": generator.get_state()}


def _restore_rng(state, generator):
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def _load_complete(directory, expected):
    path = directory / "complete.json"
    if not path.exists():
        return None
    receipt = read(path)
    if receipt.get("complete") is not True or receipt["identity"] != expected:
        raise RuntimeError("Completed fit identity changed")
    for name, wanted in receipt["files"].items():
        verify(directory / name, wanted)
    return receipt


def _finish(directory, identity, files, **details):
    value = {"complete": True, "identity": identity,
             "files": {name: sha256(directory / name) for name in files},
             "job_id": os.environ["SLURM_JOB_ID"], **details}
    atomic_json(directory / "complete.json", value)
    return value


def _identity(root, seed, stage):
    campaign(root)
    return {"campaign_sha256": sha256(Path(root) / "campaign.json"), "seed": seed, "stage": stage,
            "cls_manifest_sha256": sha256(Path(root) / "cls" / "manifest.json")}


def _resume(directory, identity, model, optimizer, generator):
    if not (directory / "resume.pt").exists():
        return None
    state = torch.load(directory / "resume.pt", map_location="cpu", weights_only=True)
    if state["identity"] != identity:
        raise RuntimeError("Resume requires identical source, roles, seed and inputs")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    _restore_rng(state["rng"], generator)
    return state


def fit_heads(root, seed):
    device = _device()
    initialize(seed)
    root = Path(root)
    directory = run_dir(root, seed) / "heads"
    identity = _identity(root, seed, "heads")
    recipe = RECIPE["head"]
    with lock(directory, "train"):
        complete = _load_complete(directory, identity)
        if complete is not None:
            return complete
        data = load_role(root, "head_train")
        labels = torch.from_numpy(data["metadata"]["label"])
        if set(labels.tolist()) != set(range(100)):
            raise RuntimeError("Head training lacks at least one ground-truth class; fixed split is not changed")
        model = LayerHeads().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
        generator = torch.Generator().manual_seed(seed)
        config = {**identity, "recipe": recipe, "parameters": sum(p.numel() for p in model.parameters()),
                  "training_role": "head_train", "label_target": "true CIFAR-100 class",
                  "selection": "fixed final epoch16", "class_counts": torch.bincount(labels, minlength=100).tolist()}
        frozen_json(directory / "config.json", config)
        state = _resume(directory, identity, model, optimizer, generator)
        history = [] if state is None else state["history"]
        began = time.monotonic()
        for epoch in range(len(history) + 1, recipe["epochs"] + 1):
            model.train()
            order = torch.randperm(len(labels), generator=generator)
            losses = torch.zeros(12, dtype=torch.float64)
            for start in range(0, len(order), recipe["batch_size"]):
                indices = order[start:start + recipe["batch_size"]]
                cls = data["cls"][indices].float().to(device)
                target = labels[indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(cls)
                # Sum means: each independent head receives its own ordinary mean CE gradient.
                per_head = torch.stack([torch.nn.functional.cross_entropy(outputs[:, layer], target)
                                        for layer in range(12)])
                per_head.sum().backward()
                if not torch.isfinite(per_head).all() or any(p.grad is None or not torch.isfinite(p.grad).all()
                                                           for p in model.parameters()):
                    raise FloatingPointError("Nonfinite or missing auxiliary-head gradients")
                optimizer.step()
                losses += per_head.detach().cpu().double() * len(indices)
            history.append({"epoch": epoch, "cross_entropy_per_layer": (losses / len(labels)).tolist()})
            atomic_torch(directory / "resume.pt", {"identity": identity, "model": _cpu_state(model),
                         "optimizer": optimizer.state_dict(), "rng": _rng_state(generator), "history": history})
            atomic_json(directory / "history.json", history)
            print(json.dumps({"stage": "heads", "seed": seed, **history[-1]}), flush=True)
            if STOP:
                raise InterruptedError("Stopped after committed epoch; resume same command")
        atomic_torch(directory / "checkpoint.pt", {"identity": identity, "state_dict": _cpu_state(model), "epoch": recipe["epochs"]})
        _portable(directory / "model.safetensors", _cpu_state(model))
        return _finish(directory, identity, ("config.json", "history.json", "checkpoint.pt", "model.safetensors", "resume.pt"),
                       epochs=recipe["epochs"], selected_epoch=recipe["epochs"], elapsed_seconds=time.monotonic() - began,
                       peak_gpu_bytes=torch.cuda.max_memory_allocated())


def load_heads(root, seed, device):
    directory = run_dir(root, seed) / "heads"
    if _load_complete(directory, _identity(root, seed, "heads")) is None:
        raise RuntimeError("Auxiliary heads must finish before probe features are built")
    model = LayerHeads().to(device)
    model.load_state_dict(torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)["state_dict"])
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def role_features(heads, data, device):
    require_slurm()
    batches = []
    for start in range(0, len(data["cls"]), 512):
        values = heads(data["cls"][start:start + 512].float().to(device))
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite auxiliary logits")
        batches.append(values.cpu().numpy())
    return build_features(np.concatenate(batches), data["logits"].numpy(), layers=RECIPE["L"], k=RECIPE["K"])


@torch.inference_mode()
def probe_scores(model, features, device, batch_size=512):
    model.eval()
    scores = torch.cat([model(features[start:start + batch_size].to(device)).flatten().cpu()
                        for start in range(0, len(features), batch_size)]).numpy().astype(np.float64)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite linear error scores")
    return scores


def fit_probe(root, seed):
    device = _device()
    initialize(seed)
    root = Path(root)
    directory = run_dir(root, seed) / "probe"
    identity = {**_identity(root, seed, "probe"),
                "heads_checkpoint_sha256": sha256(run_dir(root, seed) / "heads" / "checkpoint.pt")}
    recipe = RECIPE["probe"]
    with lock(directory, "train"):
        complete = _load_complete(directory, identity)
        if complete is not None:
            return complete
        heads = load_heads(root, seed, device)
        train, validation = load_role(root, "probe_train"), load_role(root, "probe_val")
        raw_train, raw_val = role_features(heads, train, device), role_features(heads, validation, device)
        normalizer = fit_normalizer(raw_train)
        frozen_json(directory / "normalizer.json", normalizer)
        xtrain, xval = torch.from_numpy(normalize(raw_train, normalizer)), torch.from_numpy(normalize(raw_val, normalizer))
        targets = torch.from_numpy(train["metadata"]["y"].astype(np.float32))
        yval = validation["metadata"]["y"]
        positive, negative = int(targets.sum()), len(targets) - int(targets.sum())
        if not positive or not negative or len(np.unique(yval)) != 2:
            raise RuntimeError("Both outcomes are required in probe training and validation")
        model = torch.nn.Linear(xtrain.shape[1], 1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
        generator = torch.Generator().manual_seed(seed)
        config = {**identity, "recipe": recipe, "features": feature_names(RECIPE["L"], RECIPE["K"]),
                  "normalizer_sha256": sha256(directory / "normalizer.json"), "positive_weight": negative / positive,
                  "training_role": "probe_train", "selection_role": "probe_val", "selection_metric": "average_precision",
                  "parameters": sum(p.numel() for p in model.parameters())}
        frozen_json(directory / "config.json", config)
        state = _resume(directory, identity, model, optimizer, generator)
        history = [] if state is None else state["history"]
        best_ap = -1.0 if state is None else state["best_ap"]
        best_epoch = None if state is None else state["best_epoch"]
        best = None if state is None else state["best"]
        best_scores = None if state is None else state["best_scores"]
        began = time.monotonic()
        for epoch in range(len(history) + 1, recipe["epochs"] + 1):
            model.train()
            loss_sum = 0.0
            order = torch.randperm(len(targets), generator=generator)
            for start in range(0, len(order), recipe["batch_size"]):
                indices = order[start:start + recipe["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                logits = model(xtrain[indices].to(device)).flatten()
                loss = loss_fn(logits, targets[indices].to(device))
                loss.backward()
                if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite or missing error-probe gradients")
                optimizer.step()
                loss_sum += float(loss.detach()) * len(indices)
            scores = probe_scores(model, xval, device)
            ap = float(average_precision_score(yval, scores))
            if ap > best_ap:
                best_ap, best_epoch, best = ap, epoch, _cpu_state(model)
                best_scores = torch.from_numpy(scores.copy())
            history.append({"epoch": epoch, "training_loss": loss_sum / len(targets),
                            "validation_average_precision": ap, "best_epoch": best_epoch, "best_average_precision": best_ap})
            atomic_torch(directory / "resume.pt", {"identity": identity, "model": _cpu_state(model),
                         "optimizer": optimizer.state_dict(), "rng": _rng_state(generator), "history": history,
                         "best": best, "best_epoch": best_epoch, "best_ap": best_ap, "best_scores": best_scores})
            atomic_json(directory / "history.json", history)
            print(json.dumps({"stage": "probe", "seed": seed, **history[-1]}), flush=True)
            if STOP:
                raise InterruptedError("Stopped after committed epoch; resume same command")
        model.load_state_dict(best)
        reloaded = torch.nn.Linear(xtrain.shape[1], 1).to(device)
        reloaded.load_state_dict(best)
        restored = probe_scores(reloaded, xval, device)
        np.testing.assert_allclose(restored, best_scores.numpy(), atol=1e-6, rtol=1e-6)
        atomic_torch(directory / "checkpoint.pt", {"identity": identity, "state_dict": best, "epoch": best_epoch})
        _portable(directory / "model.safetensors", best)
        atomic_npz(directory / "validation.npz", **validation["metadata"], score=restored)
        return _finish(directory, identity, ("config.json", "history.json", "normalizer.json", "checkpoint.pt", "model.safetensors", "resume.pt", "validation.npz"),
                       epochs=recipe["epochs"], selected_epoch=best_epoch, validation_average_precision=best_ap,
                       elapsed_seconds=time.monotonic() - began, peak_gpu_bytes=torch.cuda.max_memory_allocated())


def predict(root, seed):
    device = _device()
    initialize(seed)
    root = Path(root)
    evaluation_gate(root)
    directory = run_dir(root, seed) / "predictions"
    identity = {**_identity(root, seed, "predict"), "gate_sha256": sha256(root / "evaluation_gate.json")}
    with lock(directory, "predict"):
        complete = _load_complete(directory, identity)
        if complete is not None:
            return complete
        heads = load_heads(root, seed, device)
        data = load_role(root, "dev_eval")
        raw = role_features(heads, data, device)
        normalizer = read(run_dir(root, seed) / "probe" / "normalizer.json")
        features = torch.from_numpy(normalize(raw, normalizer))
        model = torch.nn.Linear(features.shape[1], 1).to(device)
        checkpoint = torch.load(run_dir(root, seed) / "probe" / "checkpoint.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        scores = probe_scores(model, features, device)
        atomic_npz(directory / "dev_eval.npz", **data["metadata"], score=scores)
        return _finish(directory, identity, ("dev_eval.npz",), selected_epoch=checkpoint["epoch"], records=len(scores))


def _stop(_signal, _frame):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("heads", "probe", "fit", "predict"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    if args.stage in ("heads", "fit"):
        fit_heads(args.root, args.seed)
    if args.stage in ("probe", "fit"):
        fit_probe(args.root, args.seed)
    if args.stage == "predict":
        predict(args.root, args.seed)


if __name__ == "__main__":
    main()
