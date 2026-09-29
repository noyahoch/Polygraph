#!/usr/bin/env python3
"""Retrain raw-logit MLP on base_val and score disjoint meta_val/test records."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polygraph.data.splits import SplitPlan
from polygraph.data.storage import GraphStore
from polygraph.training.evaluate import detector_metrics
from evaluate_output_baselines import collect


def fit(train, base_val, seed, device):
    from sklearn.metrics import roc_auc_score

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    mean = train["logits"].mean(0).astype(np.float32)
    std = (train["logits"].std(0) + 1e-6).astype(np.float32)
    x_train = torch.from_numpy(((train["logits"] - mean) / std).astype(np.float32))
    x_val = torch.from_numpy(((base_val["logits"] - mean) / std).astype(np.float32))
    y_train = torch.from_numpy(train["y"].astype(np.float32))
    model = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Dropout(.15),
                          nn.Linear(64, 32), nn.ReLU(), nn.Dropout(.15), nn.Linear(32, 1)).to(device)
    positive = float(y_train.sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        [(len(y_train) - positive) / max(positive, 1)], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best, best_auc, best_epoch, stale = None, -np.inf, 0, 0
    for epoch in range(1, 61):
        model.train()
        for idx in torch.randperm(len(x_train), generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[idx].to(device)).view(-1), y_train[idx].to(device))
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            val = model(x_val.to(device)).view(-1).cpu().numpy()
        auc = roc_auc_score(base_val["y"], val)
        if auc > best_auc + .002:
            best, best_auc, best_epoch, stale = deepcopy(model.state_dict()), auc, epoch, 0
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(best); model.eval()
    return model, mean, std, {"best_validation_auroc": float(best_auc),
                              "best_epoch": best_epoch, "stopped_epoch": epoch}


def predict(model, data, mean, std, device):
    values = torch.from_numpy(((data["logits"] - mean) / std).astype(np.float32))
    parts = []
    with torch.no_grad():
        for batch in values.split(2048):
            parts.append(model(batch.to(device)).view(-1).cpu().numpy())
    return np.concatenate(parts)


def save(path, data, score, seed, plan_hash, training):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, score=score, y=data["y"], confidence=data["confidence"],
                        margin=data["margin"], image_id=data["image_id"],
                        source_id=data["source_id"], severity=data["severity"],
                        store_index=data["store_index"],
                        method_name=np.asarray("strict_raw_logit_mlp"),
                        plan_hash=np.asarray(plan_hash), seed=np.asarray(seed))
    path.with_suffix(".json").write_text(json.dumps({"method": "strict_raw_logit_mlp",
        "seed": seed, "metrics": detector_metrics(data["y"], score), **training}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--logits-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 1, 2])
    args = parser.parse_args()
    store = GraphStore(args.store_dir)
    training = collect(SplitPlan.load(args.training_plan), store, args.logits_dir)
    evaluation = collect(SplitPlan.load(args.evaluation_plan), store, args.logits_dir)
    plan_hash = hashlib.sha256(args.evaluation_plan.read_bytes()).hexdigest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in args.seeds:
        test_path = args.out_dir / f"scores_test_seed{seed}.npz"
        val_path = args.out_dir / f"scores_val_seed{seed}.npz"
        if test_path.exists() and val_path.exists():
            print(seed, "already complete; skipping"); continue
        model, mean, std, details = fit(training["train"], training["val"], seed, device)
        val_score = predict(model, evaluation["val"], mean, std, device)
        test_score = predict(model, evaluation["test"], mean, std, device)
        save(val_path, evaluation["val"], val_score, seed, plan_hash, details)
        save(test_path, evaluation["test"], test_score, seed, plan_hash, details)
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "mean": mean, "std": std, "seed": seed, **details},
                   args.out_dir / f"model_seed{seed}.pt")
        print(seed, details, detector_metrics(evaluation["test"]["y"], test_score)["auroc"])


if __name__ == "__main__":
    main()
