"""Detector training: produces per-seed checkpoints that evaluation.py consumes.

What the POC lacked: a real validation split with early stopping on val AUROC, the best
checkpoint restored, and multi-seed runs (single-seed noise was measured at ~±0.02 AUROC).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .models import ReadoutModel, SequenceConcatModel


@dataclass
class TrainConfig:
    layers: List[int] = field(default_factory=lambda: [11])
    readout: str = "cls_gated"
    tau: Optional[float] = None  # load-time edge views; stored in the checkpoint so
    top_k: Optional[int] = None  # evaluation is guaranteed to use the same graphs
    hidden_dim: int = 32
    gnn_layers: int = 2
    dropout: float = 0.15
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 60
    patience: int = 8
    min_delta: float = 0.002
    seed: int = 7


def build_model(config: TrainConfig, in_dim: int, edge_dim: int) -> nn.Module:
    if len(config.layers) == 1:
        return ReadoutModel(in_dim, edge_dim, config.hidden_dim, config.gnn_layers,
                            config.dropout, config.readout)
    return SequenceConcatModel(in_dim, edge_dim, config.hidden_dim, config.gnn_layers,
                               config.dropout, layer_count=len(config.layers))


@torch.no_grad()
def collect(model: nn.Module, dataset, device, batch_size: int = 64) -> Dict[str, np.ndarray]:
    """Detector logits plus the metadata every metric and baseline needs."""
    from torch_geometric.loader import DataLoader

    model.eval()
    fields = ("logit", "y", "confidence", "margin", "source_id", "severity")
    out: Dict[str, list] = {f: [] for f in fields}
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        logits, _ = model(batch)
        out["logit"] += logits.cpu().tolist()
        for f in fields[1:]:
            out[f] += getattr(batch, f).view(-1).cpu().tolist()
    return {f: np.asarray(v) for f, v in out.items()}


def train_detector(config: TrainConfig, train_ds, val_ds, device) -> Tuple[nn.Module, List[dict]]:
    from sklearn.metrics import roc_auc_score
    from torch_geometric.loader import DataLoader

    random.seed(config.seed), np.random.seed(config.seed), torch.manual_seed(config.seed)
    sample = train_ds[0]
    model = build_model(config, int(sample.x.shape[1]), int(sample.edge_attr.shape[1])).to(device)
    loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    positives = sum(int(train_ds[i].y.item() > 0.5) for i in range(len(train_ds)))
    pos_weight = torch.tensor([max(len(train_ds) - positives, 1) / max(positives, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_state, best_val, best_epoch, stale, history = None, -np.inf, 0, 0, []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = seen = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.y.numel()
            seen += batch.y.numel()
        val = collect(model, val_ds, device, config.batch_size)
        val_auroc = float(roc_auc_score(val["y"], val["logit"])) if len(np.unique(val["y"])) > 1 else 0.5
        history.append(dict(epoch=epoch, train_loss=total / max(seen, 1), val_auroc=val_auroc))
        if val_auroc > best_val + config.min_delta:
            best_val, best_epoch, stale = val_auroc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or stale >= config.patience:
            print(f"  epoch {epoch}: loss {total / max(seen, 1):.4f}, val AUROC {val_auroc:.4f}", flush=True)
        if stale >= config.patience:
            print(f"  early stop (best {best_val:.4f} @ epoch {best_epoch})", flush=True)
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, history


def train_run(store_dir: Path, plan_path: Path, out_dir: Path, config: TrainConfig,
              seeds: Sequence[int], device) -> None:
    """One checkpoint per seed into out_dir; evaluation is a separate step."""
    from ..data.splits import SplitPlan
    from ..data.storage import AttentionGraphDataset, GraphStore

    plan = SplitPlan.load(plan_path)
    store = GraphStore(store_dir)
    datasets = {n: AttentionGraphDataset(store, config.layers, plan.splits[n],
                                         tau=config.tau, top_k=config.top_k) for n in ("train", "val")}
    sample = datasets["train"][0]
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        print(f"— seed {seed} —", flush=True)
        seed_config = TrainConfig(**{**asdict(config), "seed": seed})
        model, history = train_detector(seed_config, datasets["train"], datasets["val"], device)
        torch.save({"state_dict": model.state_dict(), "config": asdict(seed_config),
                    "in_dim": int(sample.x.shape[1]), "edge_dim": int(sample.edge_attr.shape[1]),
                    "plan": str(plan_path), "history": history},
                   out_dir / f"model_seed{seed}.pt")
    (out_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=2))
    print(f"checkpoints in {out_dir}", flush=True)


def load_checkpoint(path: Path, device) -> Tuple[nn.Module, TrainConfig]:
    payload = torch.load(path, map_location="cpu")
    config = TrainConfig(**payload["config"])
    model = build_model(config, payload["in_dim"], payload["edge_dim"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), config
