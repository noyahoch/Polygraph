#!/usr/bin/env python3
"""ConfidNet-style TCP regression from the frozen ViT's final stored CLS state."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_output_baselines import collect
from polygraph.data.splits import SplitPlan
from polygraph.data.storage import GraphStore
from polygraph.training.evaluate import detector_metrics


def targets(data):
    logits = data["logits"].astype(np.float64)
    logits -= logits.max(1, keepdims=True)
    probability = np.exp(logits); probability /= probability.sum(1, keepdims=True)
    tcp = probability[np.arange(len(probability)), data["label"]]
    return tcp, np.log(np.clip(tcp, 1e-6, 1 - 1e-6) / np.clip(1 - tcp, 1e-6, 1))


def train_tcp(splits, seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if "cls" not in splits["train"]:
        raise RuntimeError("graph store has no CLS embeddings")
    mean, std = splits["train"]["cls"].mean(0), splits["train"]["cls"].std(0) + 1e-6
    x = {n: torch.tensor((d["cls"] - mean) / std, dtype=torch.float32) for n, d in splits.items()}
    target = {n: torch.tensor(targets(d)[1], dtype=torch.float32) for n, d in splits.items()}
    model = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 32),
                          nn.ReLU(), nn.Dropout(.15), nn.Linear(32, 1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best, best_auc, best_epoch, stale = None, -np.inf, 0, 0
    for epoch in range(1, 61):
        model.train()
        for index in torch.randperm(len(x["train"]), generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x["train"][index].to(device)).view(-1)
            loss = nn.functional.smooth_l1_loss(prediction, target["train"][index].to(device))
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): val = model(x["val"].to(device)).view(-1).cpu().numpy()
        auc = roc_auc_score(splits["val"]["y"], -val)
        if auc > best_auc + .002:
            best_auc, best_epoch, stale = auc, epoch, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 8: break
    model.load_state_dict(best); model.eval()
    with torch.no_grad(): scores = {n: -model(v.to(device)).view(-1).cpu().numpy() for n, v in x.items()}
    return scores, {"best_validation_auroc": float(best_auc), "best_epoch": best_epoch,
                    "parameter_count": sum(p.numel() for p in model.parameters())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True); p.add_argument("--store-dir", type=Path, required=True)
    p.add_argument("--logits-dir", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 1, 2]); args = p.parse_args()
    plan, store = SplitPlan.load(args.plan), GraphStore(args.store_dir)
    splits = collect(plan, store, args.logits_dir)
    plan_hash = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in args.seeds:
        scores, training = train_tcp(splits, seed, device)
        for name, data in splits.items():
            tcp, tcp_logit = targets(data)
            predicted_tcp_logit = -scores[name]
            metadata = {"method": "cls_tcp", "seed": seed, "split": name,
                        "metrics": detector_metrics(data["y"], scores[name]), **training,
                        "tcp_mae": float(np.mean(np.abs(1 / (1 + np.exp(-predicted_tcp_logit)) - tcp))),
                        "tcp_logit_spearman": float(spearmanr(predicted_tcp_logit, tcp_logit).statistic)}
            path = args.out_dir / f"scores_{name}_seed{seed}.npz"; path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, score=scores[name], y=data["y"], confidence=data["confidence"],
                                margin=data["margin"], image_id=data["image_id"], source_id=data["source_id"],
                                severity=data["severity"], store_index=data["store_index"],
                                method_name=np.asarray("cls_tcp"), plan_hash=np.asarray(plan_hash), seed=np.asarray(seed))
            path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
