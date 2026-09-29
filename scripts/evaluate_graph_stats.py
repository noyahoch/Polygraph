#!/usr/bin/env python3
"""Fit fixed failure predictors on aligned final-layer graph statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polygraph.data.sidecars import AlignedSidecar
from polygraph.data.splits import SplitPlan
from polygraph.data.storage import GraphStore
from polygraph.training.evaluate import detector_metrics


def collect(plan, store, sidecar_dir):
    sidecar = AlignedSidecar(sidecar_dir, store.store_dir, "graph_stats", layer=11)
    result = {}
    for name, keys in plan.splits.items():
        indices = sorted(store.indices_for(keys))
        features, metadata = [], {key: [] for key in
                                  ("y", "confidence", "margin", "image_id", "source_id", "severity")}
        for index in indices:
            shard, offset = store.locate(index)
            payload, side_offset = sidecar.locate(index)
            features.append(payload["features"][side_offset].numpy())
            metadata["y"].append(float(shard.meta["y_err"][offset]))
            metadata["confidence"].append(float(shard.meta["confidence"][offset]))
            metadata["margin"].append(float(shard.meta["margin"][offset]))
            metadata["image_id"].append(int(shard.meta["base_index"][offset]))
            metadata["source_id"].append(int(shard.meta["source_id"][offset]))
            metadata["severity"].append(int(shard.meta["severity"][offset]))
        result[name] = {"x": np.stack(features).astype(np.float32),
                        **{key: np.asarray(value) for key, value in metadata.items()},
                        "store_index": np.asarray(indices, np.int64)}
    return result


def fit_classical(splits):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "graph_stats_lr": make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=7)),
        "graph_stats_hgb": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=.05, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=True, random_state=7),
    }
    scores = {}
    for name, model in models.items():
        model.fit(splits["train"]["x"], splits["train"]["y"])
        scores[name] = {split: model.predict_proba(data["x"])[:, 1]
                        for split, data in splits.items()}
    return scores


def fit_mlp(splits, seed, device):
    from sklearn.metrics import roc_auc_score

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    mean = splits["train"]["x"].mean(0)
    std = splits["train"]["x"].std(0) + 1e-6
    x = {name: torch.from_numpy((data["x"] - mean) / std) for name, data in splits.items()}
    y = {name: torch.from_numpy(data["y"].astype(np.float32)) for name, data in splits.items()}
    model = nn.Sequential(nn.Linear(x["train"].shape[1], 64), nn.ReLU(), nn.Dropout(.15),
                          nn.Linear(64, 32), nn.ReLU(), nn.Dropout(.15), nn.Linear(32, 1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    best, best_auc, stale = None, -np.inf, 0
    for _epoch in range(60):
        model.train()
        for index in torch.randperm(len(x["train"]), generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x["train"][index].to(device)).view(-1),
                           y["train"][index].to(device))
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            val = model(x["val"].to(device)).view(-1).cpu().numpy()
        auc = roc_auc_score(y["val"].numpy(), val)
        if auc > best_auc + .002:
            best_auc, stale = auc, 0
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(best); model.eval()
    with torch.no_grad():
        return ({name: model(value.to(device)).view(-1).cpu().numpy() for name, value in x.items()},
                {"best_validation_auroc": float(best_auc)})


def save(out, method, seed, split, data, score, plan_hash, selected=None):
    directory = out / method
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"scores_{split}_seed{seed}.npz"
    np.savez_compressed(path, score=score, y=data["y"], confidence=data["confidence"],
                        margin=data["margin"], image_id=data["image_id"],
                        source_id=data["source_id"], severity=data["severity"],
                        store_index=data["store_index"], method_name=np.asarray(method),
                        plan_hash=np.asarray(plan_hash), seed=np.asarray(seed))
    path.with_suffix(".json").write_text(json.dumps({
        "method": method, "seed": seed, "split": split,
        "metrics": detector_metrics(data["y"], score),
        "selected_hyperparameters": selected or {}}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 1, 2])
    args = parser.parse_args()
    plan, store = SplitPlan.load(args.plan), GraphStore(args.store_dir)
    splits = collect(plan, store, args.features_dir)
    plan_hash = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    for method, scores in fit_classical(splits).items():
        for split, data in splits.items():
            save(args.out_dir, method, 0, split, data, scores[split], plan_hash)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in args.seeds:
        scores, selected = fit_mlp(splits, seed, device)
        for split, data in splits.items():
            save(args.out_dir, "graph_stats_mlp", seed, split, data,
                 scores[split], plan_hash, selected)


if __name__ == "__main__":
    main()
