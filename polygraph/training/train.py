"""Detector training: produces per-seed checkpoints that evaluate.py consumes.

Beyond the POC: a real validation split with early stopping on val AUROC, the best state
restored, and *segmented* execution — Metal compiles one kernel variant per unique tensor
shape and never frees them, and threshold graphs give every batch a unique shape, so long
runs leak until the kernel OOM-kills the process. With `epochs_per_process` set, training
persists its full state every epoch, exits after N epochs, and resumes in a fresh process
(clean kernel cache) when relaunched; the caller loops until the final checkpoint exists.
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
    shuffle_labels: bool = False  # negative control: must score ~0.5 on test
    charm: bool = False  # CHARM-lite: one union graph per image, L*H-dim edge features
    hidden: bool = False  # variant 2: per-token hidden states appended to node features
    epochs_per_process: Optional[int] = None  # segment length; None disables


class ShardShuffleSampler(torch.utils.data.Sampler):
    """Shuffles shard blocks, then items within each block. Full random shuffling over
    ~5 GB shards with a 2-shard cache degenerates to one multi-gigabyte load per sample;
    block shuffling keeps disk access sequential while store order is already random
    (plans shuffle their keys), so SGD still sees a fresh permutation every epoch."""

    def __init__(self, dataset, seed: int):
        self.blocks, self.seed, self.epoch = dataset.shard_blocks(), seed, 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        order = list(self.blocks)
        rng.shuffle(order)
        for block in order:
            block = list(block)
            rng.shuffle(block)
            yield from block

    def __len__(self):
        return sum(len(block) for block in self.blocks)


class _ShuffledLabels(torch.utils.data.Dataset):
    """Permutes training labels (labels only; graphs untouched). A pipeline with any
    leak lets a model score above chance here; a clean one cannot."""

    def __init__(self, dataset, seed: int):
        self.dataset = dataset
        permutation = np.random.default_rng(seed).permutation(len(dataset))
        self._labels = dataset.labels()[permutation]
        self.shard_blocks = dataset.shard_blocks

    def labels(self) -> np.ndarray:
        return self._labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample.y = torch.tensor([self._labels[index]], dtype=torch.float32)
        return sample


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
    if device.type == "mps":
        torch.mps.empty_cache()
    return {f: np.asarray(v) for f, v in out.items()}


def train_detector(config: TrainConfig, train_ds, val_ds, device,
                   state_path: Optional[Path] = None) -> Tuple[nn.Module, List[dict], bool]:
    """Returns (model, history, finished). With epochs_per_process set, runs at most that
    many new epochs, persists full state to state_path, and returns finished=False so the
    caller can exit the process (resetting Metal's kernel cache) and relaunch to resume."""
    from sklearn.metrics import roc_auc_score
    from torch_geometric.loader import DataLoader

    random.seed(config.seed), np.random.seed(config.seed), torch.manual_seed(config.seed)
    sample = train_ds[0]
    model = build_model(config, int(sample.x.shape[1]), int(sample.edge_attr.shape[1])).to(device)
    if hasattr(train_ds, "shard_blocks"):
        sampler = ShardShuffleSampler(train_ds, config.seed)
        loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler)
        positives = float(train_ds.labels().sum())
    else:  # in-memory test datasets
        sampler = None
        loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        positives = sum(int(train_ds[i].y.item() > 0.5) for i in range(len(train_ds)))
    pos_weight = torch.tensor([max(len(train_ds) - positives, 1) / max(positives, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_state, best_val, best_epoch, stale, history = None, -np.inf, 0, 0, []
    start_epoch = 1
    if state_path is not None and state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best_state, best_val = state["best_state"], state["best_val"]
        best_epoch, stale, history = state["best_epoch"], state["stale"], state["history"]
        start_epoch = state["epoch"] + 1
        if sampler is not None:
            sampler.epoch = state["epoch"]  # keep the per-epoch permutation sequence
        print(f"  resuming from epoch {start_epoch} (best {best_val:.4f} @ {best_epoch})", flush=True)

    finished = True
    epochs_this_process = 0
    for epoch in range(start_epoch, config.epochs + 1):
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
        if device.type == "mps":
            torch.mps.empty_cache()
        val_auroc = float(roc_auc_score(val["y"], val["logit"])) if len(np.unique(val["y"])) > 1 else 0.5
        history.append(dict(epoch=epoch, train_loss=total / max(seen, 1), val_auroc=val_auroc))
        if val_auroc > best_val + config.min_delta:
            best_val, best_epoch, stale = val_auroc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or stale >= config.patience:
            print(f"  epoch {epoch}: loss {total / max(seen, 1):.4f}, val AUROC {val_auroc:.4f}", flush=True)
        if state_path is not None:
            tmp = state_path.with_suffix(".tmp")
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "best_state": best_state,
                        "best_val": best_val, "best_epoch": best_epoch, "stale": stale,
                        "history": history}, tmp)
            tmp.rename(state_path)
        if stale >= config.patience:
            print(f"  early stop (best {best_val:.4f} @ epoch {best_epoch})", flush=True)
            break
        epochs_this_process += 1
        if (config.epochs_per_process and epochs_this_process >= config.epochs_per_process
                and epoch < config.epochs):
            print(f"  segment done at epoch {epoch}; exiting to reset Metal cache", flush=True)
            finished = False
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, history, finished


def train_run(store_dir: Path, plan_path: Path, out_dir: Path, config: TrainConfig,
              seeds: Sequence[int], device) -> None:
    """One checkpoint per seed into out_dir; evaluation is a separate step. At most one
    seed segment runs per process invocation — the caller loops until all checkpoints
    exist, which also gives every seed a clean Metal cache."""
    from ..data.splits import SplitPlan
    from ..data.storage import AttentionGraphDataset, GraphStore

    plan = SplitPlan.load(plan_path)
    store = GraphStore(store_dir)
    if config.charm:
        from ..data.storage import CharmDataset

        datasets = {n: CharmDataset(store, plan.splits[n], tau=config.tau)
                    for n in ("train", "val")}
    else:
        hidden_dir = store_dir.parent / "hidden12" if config.hidden else None
        datasets = {n: AttentionGraphDataset(store, config.layers, plan.splits[n],
                                             tau=config.tau, top_k=config.top_k,
                                             hidden_dir=hidden_dir) for n in ("train", "val")}
    if config.shuffle_labels:
        datasets["train"] = _ShuffledLabels(datasets["train"], seed=999)
    sample = datasets["train"][0]
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        final_path = out_dir / f"model_seed{seed}.pt"
        if final_path.exists():
            continue
        print(f"— seed {seed} —", flush=True)
        seed_config = TrainConfig(**{**asdict(config), "seed": seed})
        state_path = out_dir / f"state_seed{seed}.pt"
        model, history, finished = train_detector(seed_config, datasets["train"],
                                                  datasets["val"], device, state_path)
        if finished:
            torch.save({"state_dict": model.state_dict(), "config": asdict(seed_config),
                        "in_dim": int(sample.x.shape[1]), "edge_dim": int(sample.edge_attr.shape[1]),
                        "plan": str(plan_path), "history": history}, final_path)
            state_path.unlink(missing_ok=True)
            print(f"checkpoint saved: {final_path}", flush=True)
        return  # one segment per process, finished or not
    (out_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=2))
    print(f"all requested checkpoints present in {out_dir}", flush=True)


def load_checkpoint(path: Path, device) -> Tuple[nn.Module, TrainConfig]:
    payload = torch.load(path, map_location="cpu")
    config = TrainConfig(**payload["config"])
    model = build_model(config, payload["in_dim"], payload["edge_dim"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), config
