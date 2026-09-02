#!/usr/bin/env python3
"""Validation-selected full-logit failure-prediction baselines."""

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
from polygraph.training.research_eval import P_NORMS, pnorm_error_score, select_pnorm


def collect(plan: SplitPlan, store: GraphStore, logits_dir: Path):
    sidecar = AlignedSidecar(logits_dir, store.store_dir, "logits")
    split_indices = {name: sorted(store.indices_for(keys)) for name, keys in plan.splits.items()}
    union = sorted(set().union(*map(set, split_indices.values())))
    rows = {}
    for index in union:
        shard, offset = store.locate(index)
        logit_shard, logit_offset = sidecar.locate(index)
        rows[index] = (logit_shard["logits"][logit_offset].float().numpy(),
                       float(shard.meta["y_err"][offset]), float(shard.meta["confidence"][offset]),
                       float(shard.meta["margin"][offset]), int(shard.meta["base_index"][offset]),
                       int(shard.meta["source_id"][offset]), int(shard.meta["severity"][offset]))
    result = {}
    for name, indices in split_indices.items():
        values = [rows[i] for i in indices]
        result[name] = {"logits": np.stack([x[0] for x in values]),
                        "y": np.asarray([x[1] for x in values], np.int8),
                        "confidence": np.asarray([x[2] for x in values], np.float32),
                        "margin": np.asarray([x[3] for x in values], np.float32),
                        "image_id": np.asarray([x[4] for x in values], np.int32),
                        "source_id": np.asarray([x[5] for x in values], np.int16),
                        "severity": np.asarray([x[6] for x in values], np.int8),
                        "store_index": np.asarray(indices, np.int64)}
    return result


def deterministic_scores(data, selected_p, selected_softmax):
    logits = data["logits"].astype(np.float64)
    shifted = logits - logits.max(1, keepdims=True)
    probability = np.exp(shifted); probability /= probability.sum(1, keepdims=True)
    entropy = -(probability * np.log(np.maximum(probability, 1e-300))).sum(1)
    logsumexp = logits.max(1) + np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1))
    return {"msp": 1 - data["confidence"], "margin": -data["margin"], "entropy": entropy,
            "energy": -logsumexp, "max_logit": -logits.max(1),
            "p_normalized": pnorm_error_score(logits, selected_p["p"]),
            "pnorm_softmax": pnorm_error_score(logits, selected_softmax["p"],
                                                selected_softmax["temperature"])}


def logistic_scores(splits, transform, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=seed))
    model.fit(transform(splits["train"]["logits"]), splits["train"]["y"])
    return {name: model.predict_proba(transform(data["logits"]))[:, 1]
            for name, data in splits.items()}


def mlp_scores(splits, transform, seed, device):
    from sklearn.metrics import roc_auc_score

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    raw = {name: transform(data["logits"]).astype(np.float32) for name, data in splits.items()}
    mean, std = raw["train"].mean(0), raw["train"].std(0) + 1e-6
    x = {name: torch.from_numpy((value - mean) / std) for name, value in raw.items()}
    y = {name: torch.from_numpy(data["y"].astype(np.float32)) for name, data in splits.items()}
    model = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Dropout(.15), nn.Linear(64, 32),
                          nn.ReLU(), nn.Dropout(.15), nn.Linear(32, 1)).to(device)
    positives = float(y["train"].sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        [(len(y["train"]) - positives) / max(positives, 1)], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best, best_auc, stale = None, -np.inf, 0
    generator = torch.Generator().manual_seed(seed)
    for _epoch in range(60):
        model.train()
        for indices in torch.randperm(len(x["train"]), generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x["train"][indices].to(device)).view(-1),
                           y["train"][indices].to(device))
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            score = model(x["val"].to(device)).view(-1).cpu().numpy()
        auc = roc_auc_score(y["val"].numpy(), score)
        if auc > best_auc + .002:
            best_auc, stale = auc, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 8: break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        return {name: model(value.to(device)).view(-1).cpu().numpy() for name, value in x.items()}


def save_scores(out: Path, method: str, seed: int, split_name: str, data, score,
                plan_hash: str, selected: dict):
    path = out / method / f"scores_{split_name}_seed{seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, score=np.asarray(score), y=data["y"], confidence=data["confidence"],
                        margin=data["margin"], image_id=data["image_id"], source_id=data["source_id"],
                        severity=data["severity"], store_index=data["store_index"],
                        method_name=np.asarray(method), plan_hash=np.asarray(plan_hash), seed=np.asarray(seed))
    metrics = detector_metrics(data["y"], np.asarray(score))
    path.with_suffix(".json").write_text(json.dumps(
        {"method": method, "seed": seed, "split": split_name, "metrics": metrics,
         "selected_hyperparameters": selected}, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True); p.add_argument("--store-dir", type=Path, required=True)
    p.add_argument("--logits-dir", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 1, 2])
    args = p.parse_args()
    plan, store = SplitPlan.load(args.plan), GraphStore(args.store_dir)
    splits = collect(plan, store, args.logits_dir)
    plan_hash = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    selected_p = select_pnorm(splits["val"]["logits"], splits["val"]["y"])
    selected_softmax = select_pnorm(splits["val"]["logits"], splits["val"]["y"],
                                    temperatures=(.25, .5, 1, 2, 4))
    deterministic = {name: deterministic_scores(data, selected_p, selected_softmax)
                     for name, data in splits.items()}
    for method in deterministic["train"]:
        for split_name, data in splits.items():
            save_scores(args.out_dir, method, 0, split_name, data, deterministic[split_name][method],
                        plan_hash, selected_p if method == "p_normalized" else
                        selected_softmax if method == "pnorm_softmax" else {})
    centered = lambda z: np.sort(z - z.mean(1, keepdims=True), axis=1)[:, ::-1].copy()
    transforms = {"raw_logit_lr": lambda z: z, "sorted_centered_logit_lr": centered}
    for seed in args.seeds:
        for method, transform in transforms.items():
            scores = logistic_scores(splits, transform, seed)
            for split_name, data in splits.items():
                save_scores(args.out_dir, method, seed, split_name, data, scores[split_name], plan_hash, {})
        for method, transform in (("raw_logit_mlp", lambda z: z),
                                  ("sorted_centered_logit_mlp", centered)):
            scores = mlp_scores(splits, transform, seed,
                                torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            for split_name, data in splits.items():
                save_scores(args.out_dir, method, seed, split_name, data, scores[split_name], plan_hash, {})


if __name__ == "__main__":
    main()
