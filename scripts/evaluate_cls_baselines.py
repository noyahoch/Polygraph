#!/usr/bin/env python3
"""Current-plan final-CLS and CLS-trajectory baselines from the existing store."""

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

from polygraph.data.splits import SplitPlan
from polygraph.data.storage import GraphStore
from polygraph.training.evaluate import detector_metrics


class GRUHead(nn.Module):
    def __init__(self, dim=768, hidden=128):
        super().__init__()
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(.15), nn.Linear(hidden, 1))

    def forward(self, x):
        _, state = self.gru(x)
        return self.head(state[-1]).view(-1)


def collect(plan, store):
    result = {}
    for split, keys in plan.splits.items():
        indices = sorted(store.indices_for(keys))
        cls, metadata = [], {name: [] for name in
                             ("y", "confidence", "margin", "image_id", "source_id", "severity")}
        for index in indices:
            shard, offset = store.locate(index)
            if shard.cls_embeddings is None:
                raise RuntimeError("graph store lacks CLS trajectories")
            cls.append(shard.cls_embeddings[offset].float().numpy())
            metadata["y"].append(float(shard.meta["y_err"][offset]))
            metadata["confidence"].append(float(shard.meta["confidence"][offset]))
            metadata["margin"].append(float(shard.meta["margin"][offset]))
            metadata["image_id"].append(int(shard.meta["base_index"][offset]))
            metadata["source_id"].append(int(shard.meta["source_id"][offset]))
            metadata["severity"].append(int(shard.meta["severity"][offset]))
        result[split] = {"cls": np.stack(cls).astype(np.float32),
                         **{k: np.asarray(v) for k, v in metadata.items()},
                         "store_index": np.asarray(indices, np.int64)}
    return result


def fit(splits, method, seed, device):
    from sklearn.metrics import roc_auc_score
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    raw = {k: (v["cls"][:, -1] if method == "cls_mlp" else v["cls"])
           for k, v in splits.items()}
    flat = raw["train"].reshape(len(raw["train"]), -1)
    mean, std = flat.mean(0), flat.std(0) + 1e-6
    x = {k: torch.from_numpy(((v.reshape(len(v), -1) - mean) / std)
                             .reshape(v.shape).astype(np.float32)) for k, v in raw.items()}
    y = {k: torch.from_numpy(v["y"].astype(np.float32)) for k, v in splits.items()}
    model = (nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 1))
             if method == "cls_mlp" else GRUHead()).to(device)
    positive = float(y["train"].sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        [(len(y["train"]) - positive) / max(positive, 1)], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best, best_auc, best_epoch, stale = None, -np.inf, 0, 0
    for epoch in range(1, 61):
        model.train()
        for idx in torch.randperm(len(x["train"]), generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x["train"][idx].to(device)).view(-1), y["train"][idx].to(device))
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            score = model(x["val"].to(device)).view(-1).cpu().numpy()
        auc = roc_auc_score(y["val"].numpy(), score)
        if auc > best_auc + .002:
            best, best_auc, best_epoch, stale = ({k: v.detach().cpu().clone()
                                                 for k, v in model.state_dict().items()}, auc, epoch, 0)
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(best); model.eval()
    with torch.no_grad():
        scores = {k: model(v.to(device)).view(-1).cpu().numpy() for k, v in x.items()}
    return scores, {"best_validation_auroc": float(best_auc), "best_epoch": best_epoch,
                    "parameter_count": sum(p.numel() for p in model.parameters())}


def save(out, method, seed, split, data, score, plan_hash, training):
    directory = out / method; directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"scores_{split}_seed{seed}.npz"
    np.savez_compressed(path, score=score, y=data["y"], confidence=data["confidence"],
                        margin=data["margin"], image_id=data["image_id"],
                        source_id=data["source_id"], severity=data["severity"],
                        store_index=data["store_index"], method_name=np.asarray(method),
                        plan_hash=np.asarray(plan_hash), seed=np.asarray(seed))
    path.with_suffix(".json").write_text(json.dumps({"method": method, "seed": seed,
        "split": split, "metrics": detector_metrics(data["y"], score), **training}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 1, 2])
    args = parser.parse_args()
    plan, store = SplitPlan.load(args.plan), GraphStore(args.store_dir)
    splits = collect(plan, store)
    plan_hash = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for method in ("cls_mlp", "cls_seq"):
        for seed in args.seeds:
            scores, training = fit(splits, method, seed, device)
            for split, data in splits.items():
                save(args.out_dir, method, seed, split, data, scores[split], plan_hash, training)
            print(method, seed, detector_metrics(splits["test"]["y"], scores["test"])["auroc"])


if __name__ == "__main__":
    main()
