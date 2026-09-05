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
    architecture: str = "transformerconv"
    node_features: str = "base"
    edge_features: str = "attention"
    logits_dir: Optional[str] = None
    message_stats_dir: Optional[str] = None
    compact_evidence_dir: Optional[str] = None
    rewire_mode: str = "none"
    temporal_edges: bool = False
    tcp_multitask: bool = False
    jumping_knowledge: bool = False
    multilayer_mode: str = "none"
    multilayer_family: str = "edge_gated_mean"
    rewire_cache_dir: Optional[str] = None


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
    architecture = getattr(config, "architecture", "transformerconv")
    model = None
    if architecture == "edge_set" or config.readout == "edge_set":
        from .models import EdgeSetModel

        model = EdgeSetModel(in_dim, edge_dim, config.hidden_dim, config.dropout)
    elif architecture == "node_edge_set":
        from .models import NodeEdgeSetModel

        model = NodeEdgeSetModel(in_dim, edge_dim, config.hidden_dim, config.dropout)
    elif architecture == "endpoint_set":
        from .models import EndpointSetModel

        model = EndpointSetModel(in_dim, edge_dim, config.hidden_dim, config.dropout)
    elif architecture == "hidden_token_set":
        from .models import HiddenTokenSetModel
        model = HiddenTokenSetModel(in_dim, config.hidden_dim, config.dropout)
    elif architecture == "m5_node_edge_set":
        from .models import M5NodeEdgeSetModel
        model = M5NodeEdgeSetModel(in_dim, edge_dim, config.hidden_dim, config.dropout)
    elif architecture == "m5_endpoint_set":
        from .models import M5EndpointSetModel
        model = M5EndpointSetModel(in_dim, edge_dim, config.hidden_dim, config.dropout)
    elif architecture in {"transformerconv_residual", "gine", "edge_gated_mean", "gatv2"}:
        from .models import ResidualGraphModel
        model = ResidualGraphModel(in_dim, edge_dim, config.hidden_dim, config.gnn_layers,
                                   config.dropout, architecture,
                                   getattr(config, "jumping_knowledge", False))
    elif architecture == "last4_token_set":
        from .models import LastFourTokenSetModel
        model = LastFourTokenSetModel(in_dim, config.hidden_dim, config.dropout)
    elif architecture == "last4_union_graph":
        from .models import LastFourUnionGraphModel
        model = LastFourUnionGraphModel(in_dim, edge_dim, config.hidden_dim, config.gnn_layers,
                                        config.dropout, config.multilayer_family)
    elif architecture == "last4_union_endpoint_set":
        from .models import LastFourUnionEndpointSetModel
        model = LastFourUnionEndpointSetModel(in_dim, edge_dim, config.hidden_dim, config.dropout)
    elif architecture == "simple_mpnn":
        from .models import SimpleMPNN

        model = SimpleMPNN(in_dim, edge_dim, config.hidden_dim, config.gnn_layers,
                           config.dropout, config.readout)
    elif architecture == "temporal_mpnn":
        from .models import TemporalMPNN

        model = TemporalMPNN(in_dim, edge_dim, config.hidden_dim, config.gnn_layers,
                             config.dropout, config.readout)
    elif architecture != "transformerconv":
        raise ValueError(f"unknown architecture: {architecture}")
    elif len(config.layers) == 1:
        model = ReadoutModel(in_dim, edge_dim, config.hidden_dim, config.gnn_layers,
                             config.dropout, config.readout)
    else:
        model = SequenceConcatModel(in_dim, edge_dim, config.hidden_dim, config.gnn_layers,
                                    config.dropout, layer_count=len(config.layers))
    if getattr(config, "tcp_multitask", False):
        if config.readout != "cls_gated" or architecture not in {"transformerconv", "simple_mpnn"}:
            raise ValueError("TCP multitask currently requires cls_gated transformerconv/simple_mpnn")
        from .models import TCPMultiTaskModel
        model = TCPMultiTaskModel(model, 2 * config.hidden_dim)
    return model


@torch.no_grad()
def collect(model: nn.Module, dataset, device, batch_size: int = 64,
            include_alignment: bool = False) -> Dict[str, np.ndarray]:
    """Detector logits plus the metadata every metric and baseline needs."""
    from torch_geometric.loader import DataLoader

    model.eval()
    # Preserve the historical public return schema by default. Research score
    # files opt into the extra stable identifiers needed for alignment checks.
    fields = ("logit", "y", "confidence", "margin", "source_id", "severity")
    if include_alignment:
        fields += ("image_id", "store_index")
    out: Dict[str, list] = {f: [] for f in fields}
    collect_tcp = hasattr(model, "forward_multitask")
    if collect_tcp:
        out["tcp_logit"] = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        if collect_tcp:
            logits, tcp_logit, _ = model.forward_multitask(batch)
            out["tcp_logit"] += tcp_logit.cpu().tolist()
        else:
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
    model = build_model(config, int(sample.x.shape[-1]), int(sample.edge_attr.shape[-1])).to(device)
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
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
        state = torch.load(state_path, map_location="cpu", weights_only=False)
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
            if config.tcp_multitask:
                logits, tcp_logit, _ = model.forward_multitask(batch)
                if not hasattr(batch, "tcp_target"):
                    raise RuntimeError("TCP target is unavailable in the training dataset")
                loss = criterion(logits, batch.y.view(-1)) + 0.2 * nn.functional.smooth_l1_loss(
                    tcp_logit, batch.tcp_target.view(-1))
            else:
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
    if getattr(config, "multilayer_mode", "none") != "none":
        from ..data.storage import LastFourGraphDataset
        datasets = {n: LastFourGraphDataset(store, plan.splits[n], config.multilayer_mode)
                    for n in ("train", "val")}
    elif config.charm:
        from ..data.storage import CharmDataset

        datasets = {n: CharmDataset(store, plan.splits[n], tau=config.tau)
                    for n in ("train", "val")}
    else:
        node_features = getattr(config, "node_features", "base")
        if config.hidden and node_features == "base":  # legacy --hidden behavior
            node_features = "hidden"
        hidden_dir = store_dir.parent / "hidden12" if node_features == "hidden" else None
        if node_features not in {"base", "hidden", "compact_evidence"}:
            raise ValueError(f"unknown node feature mode: {node_features}")
        compact_dir = (Path(config.compact_evidence_dir) if config.compact_evidence_dir else
                       store_dir.parent / "sidecars/compact_evidence_l12") \
                      if node_features == "compact_evidence" else None
        message_dir = (Path(config.message_stats_dir) if config.message_stats_dir else
                       store_dir.parent / "sidecars/message_stats_l11") \
                      if getattr(config, "edge_features", "attention") != "attention" else None
        logits_dir = (Path(config.logits_dir) if config.logits_dir else
                      store_dir.parent / "sidecars/logits") if config.tcp_multitask else None
        datasets = {n: AttentionGraphDataset(store, config.layers, plan.splits[n],
                                             tau=config.tau, top_k=config.top_k,
                                             hidden_dir=hidden_dir,
                                             rewire_mode=getattr(config, "rewire_mode", "none"),
                                             edge_features=getattr(config, "edge_features", "attention"),
                                             message_stats_dir=message_dir,
                                             compact_evidence_dir=compact_dir,
                                             temporal_edges=getattr(config, "temporal_edges", False),
                                             logits_dir=logits_dir,
                                             tcp_target=config.tcp_multitask,
                                             rewire_cache_dir=Path(config.rewire_cache_dir)
                                             if config.rewire_cache_dir else None)
                    for n in ("train", "val")}
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
            payload = {"state_dict": model.state_dict(), "config": asdict(seed_config),
                       "in_dim": int(sample.x.shape[-1]), "edge_dim": int(sample.edge_attr.shape[-1]),
                       "plan": str(plan_path), "history": history}
            temporary = final_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(final_path)
            state_path.unlink(missing_ok=True)
            print(f"checkpoint saved: {final_path}", flush=True)
        return  # one segment per process, finished or not
    (out_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=2))
    print(f"all requested checkpoints present in {out_dir}", flush=True)


def load_checkpoint(path: Path, device) -> Tuple[nn.Module, TrainConfig]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = TrainConfig(**payload["config"])
    model = build_model(config, payload["in_dim"], payload["edge_dim"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), config
