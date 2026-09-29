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


class SequenceTCP(nn.Module):
    """TCP regressor over the deployment-available twelve-layer CLS trajectory."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(768, 128, batch_first=True)
        self.head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(.15), nn.Linear(32, 1))

    def forward(self, x):
        _, state = self.gru(x)
        return self.head(state[-1]).view(-1)


def targets(data):
    logits = data["logits"].astype(np.float64)
    logits -= logits.max(1, keepdims=True)
    probability = np.exp(logits); probability /= probability.sum(1, keepdims=True)
    tcp = probability[np.arange(len(probability)), data["label"]]
    return tcp, np.log(np.clip(tcp, 1e-6, 1 - 1e-6) / np.clip(1 - tcp, 1e-6, 1))


def train_tcp(splits, seed, device, sequence=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if "cls" not in splits["train"]:
        raise RuntimeError("graph store has no CLS embeddings")
    raw = {n: d["cls"] for n, d in splits.items()}
    mean, std = raw["train"].mean(0), raw["train"].std(0) + 1e-6
    x = {n: torch.tensor((value - mean) / std, dtype=torch.float32) for n, value in raw.items()}
    target = {n: torch.tensor(targets(d)[1], dtype=torch.float32) for n, d in splits.items()}
    model = (SequenceTCP() if sequence else nn.Sequential(
        nn.Linear(768, 128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 32),
        nn.ReLU(), nn.Dropout(.15), nn.Linear(32, 1))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best, best_auc, best_epoch, stale = None, -np.inf, 0, 0
    for epoch in range(1, 61):
        model.train()
        batch_size = 128 if sequence else 256
        for index in torch.randperm(len(x["train"]), generator=generator).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x["train"][index].to(device)).view(-1)
            loss = nn.functional.smooth_l1_loss(prediction, target["train"][index].to(device))
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            val = torch.cat([model(batch.to(device)).view(-1).cpu()
                             for batch in x["val"].split(1024)]).numpy()
        auc = roc_auc_score(splits["val"]["y"], -val)
        if auc > best_auc + .002:
            best_auc, best_epoch, stale = auc, epoch, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 8: break
    model.load_state_dict(best); model.eval()
    with torch.no_grad():
        scores = {n: -torch.cat([model(batch.to(device)).view(-1).cpu()
                                 for batch in value.split(1024)]).numpy()
                  for n, value in x.items()}
    return scores, {"best_validation_auroc": float(best_auc), "best_epoch": best_epoch,
                    "parameter_count": sum(p.numel() for p in model.parameters())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True); p.add_argument("--store-dir", type=Path, required=True)
    p.add_argument("--logits-dir", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 1, 2])
    p.add_argument("--sequence", action="store_true",
                   help="regress TCP from the full 12-layer CLS trajectory with a GRU")
    args = p.parse_args()
    plan, store = SplitPlan.load(args.plan), GraphStore(args.store_dir)
    splits = collect(plan, store, args.logits_dir)
    if args.sequence:
        from evaluate_cls_baselines import collect as collect_cls

        trajectories = collect_cls(plan, store)
        for name in splits:
            if not np.array_equal(splits[name]["store_index"], trajectories[name]["store_index"]):
                raise RuntimeError(f"CLS/logit row misalignment in {name}")
            splits[name]["cls"] = trajectories[name]["cls"]
    plan_hash = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in args.seeds:
        method = "cls_seq_tcp" if args.sequence else "cls_tcp"
        expected = [args.out_dir / f"scores_{name}_seed{seed}.npz" for name in splits]
        if all(path.exists() for path in expected):
            print(method, seed, "already complete; skipping")
            continue
        scores, training = train_tcp(splits, seed, device, sequence=args.sequence)
        for name, data in splits.items():
            tcp, tcp_logit = targets(data)
            predicted_tcp_logit = -scores[name]
            metadata = {"method": method, "seed": seed, "split": name,
                        "metrics": detector_metrics(data["y"], scores[name]), **training,
                        "tcp_mae": float(np.mean(np.abs(1 / (1 + np.exp(-predicted_tcp_logit)) - tcp))),
                        "tcp_logit_spearman": float(spearmanr(predicted_tcp_logit, tcp_logit).statistic)}
            path = args.out_dir / f"scores_{name}_seed{seed}.npz"; path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, score=scores[name], y=data["y"], confidence=data["confidence"],
                                margin=data["margin"], image_id=data["image_id"], source_id=data["source_id"],
                                severity=data["severity"], store_index=data["store_index"],
                                method_name=np.asarray(method), plan_hash=np.asarray(plan_hash), seed=np.asarray(seed))
            path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
