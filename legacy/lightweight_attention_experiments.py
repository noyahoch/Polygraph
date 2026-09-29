#!/usr/bin/env python3
"""Lightweight ViT attention-graph experiments for CIFAR-100 failure prediction.

This runner keeps the original POC script intact and adds a smaller experiment
path centered on fixed top-global attention edges per ViT layer. It uses the
same frozen ViT and local CIFAR-100 parquet data as poc_gnn_vit_cifar100.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import TransformerConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax
from tqdm import tqdm

from poc_gnn_vit_cifar100 import (
    PROJECT_ROOT,
    balanced_train_test_indices,
    choose_device,
    collate_images,
    confidence_baselines,
    load_local_hf_parquet_cifar100,
    load_vit,
    pick_model,
    preprocess_batch,
    read_scan_cache,
    scan_vit_correctness,
    seed_everything,
    write_scan_cache,
)


@dataclass(frozen=True)
class SplitIndices:
    train: List[int]
    val: List[int]
    test: List[int]


class SparseGraphCache:
    def __init__(self, rows: Sequence[Dict[str, object]], layer_count: int):
        self.rows = list(rows)
        self.layer_count = layer_count
        self.by_index = {int(row["image_id"]): row for row in self.rows}

    def select(self, indices: Sequence[int]) -> List[Dict[str, object]]:
        return [self.by_index[int(index)] for index in indices]


class LayerGraphDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, object]],
        layer_positions: Sequence[int],
        layer_count: int,
        duplicate_final: bool = False,
    ):
        self.rows = list(rows)
        self.layer_positions = list(layer_positions)
        self.layer_count = layer_count
        self.duplicate_final = duplicate_final

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Data:
        row = self.rows[idx]
        graphs = row["layers"]
        assert isinstance(graphs, list)
        xs: List[Tensor] = []
        edge_indices: List[Tensor] = []
        edge_attrs: List[Tensor] = []
        layer_ids: List[Tensor] = []
        node_offset = 0

        for position, source_layer in enumerate(self.layer_positions):
            graph = graphs[-1] if self.duplicate_final else graphs[source_layer]
            assert isinstance(graph, dict)
            x = graph["x"].clone()
            assert isinstance(x, Tensor)
            if len(self.layer_positions) > 1:
                x[:, 3] = position / max(len(self.layer_positions) - 1, 1)
            edge_index = graph["edge_index"]
            edge_attr = graph["edge_attr"]
            assert isinstance(edge_index, Tensor)
            assert isinstance(edge_attr, Tensor)
            xs.append(x.float())
            edge_indices.append(edge_index.long() + node_offset)
            edge_attrs.append(edge_attr.float())
            layer_ids.append(torch.full((x.shape[0],), position, dtype=torch.long))
            node_offset += x.shape[0]

        return Data(
            x=torch.cat(xs, dim=0),
            edge_index=torch.cat(edge_indices, dim=1),
            edge_attr=torch.cat(edge_attrs, dim=0),
            layer_id=torch.cat(layer_ids, dim=0),
            y=torch.tensor([float(row["y_err"])], dtype=torch.float32),
            image_id=torch.tensor([int(row["image_id"])], dtype=torch.long),
            vit_correct=torch.tensor([float(row["vit_correct"])], dtype=torch.float32),
            confidence=torch.tensor([float(row["confidence"])], dtype=torch.float32),
        )


def normalized_patch_features(num_tokens: int, attention: Tensor, layer_idx: int, layer_count: int) -> Tensor:
    num_patches = num_tokens - 1
    side = int(math.sqrt(num_patches))
    coords = torch.zeros((num_tokens, 4), dtype=torch.float32)
    coords[0, 2] = 1.0
    coords[:, 3] = layer_idx / max(layer_count - 1, 1)
    if side * side == num_patches:
        patch = 1
        denom = max(side - 1, 1)
        for row in range(side):
            for col in range(side):
                coords[patch, 0] = row / denom
                coords[patch, 1] = col / denom
                patch += 1
    diag_by_head = torch.diagonal(attention, dim1=1, dim2=2).transpose(0, 1).contiguous()
    return torch.cat([coords, diag_by_head], dim=1)


def build_top_global_graph(attention: Tensor, edges_per_layer: int, layer_idx: int, layer_count: int) -> Dict[str, Tensor]:
    heads, num_tokens, _ = attention.shape
    max_attention = attention.max(dim=0).values.clone()
    max_attention.fill_diagonal_(-float("inf"))
    edge_count = min(edges_per_layer, num_tokens * (num_tokens - 1))
    flat_indices = torch.topk(max_attention.reshape(-1), k=edge_count).indices
    query_i = torch.div(flat_indices, num_tokens, rounding_mode="floor")
    key_j = flat_indices.remainder(num_tokens)
    edge_index = torch.stack([key_j, query_i], dim=0).long()
    edge_attr = attention[:, query_i, key_j].transpose(0, 1).contiguous().float()
    x = normalized_patch_features(num_tokens, attention, layer_idx=layer_idx, layer_count=layer_count).float()
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}


def make_fixed_splits(
    correct_indices: Sequence[int],
    wrong_indices: Sequence[int],
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    seed: int,
) -> SplitIndices:
    train_val, test = balanced_train_test_indices(
        correct_indices,
        wrong_indices,
        train_per_class + val_per_class,
        test_per_class,
        seed,
    )
    rng = random.Random(seed + 17)
    correct_lookup = set(correct_indices)
    wrong_lookup = set(wrong_indices)
    correct_train_val = [idx for idx in train_val if idx in correct_lookup]
    wrong_train_val = [idx for idx in train_val if idx in wrong_lookup]
    rng.shuffle(correct_train_val)
    rng.shuffle(wrong_train_val)
    train = correct_train_val[:train_per_class] + wrong_train_val[:train_per_class]
    val = correct_train_val[train_per_class : train_per_class + val_per_class] + wrong_train_val[
        train_per_class : train_per_class + val_per_class
    ]
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return SplitIndices(train=train, val=val, test=list(test))


def load_or_create_splits(args: argparse.Namespace, correct_indices: Sequence[int], wrong_indices: Sequence[int]) -> SplitIndices:
    split_path = PROJECT_ROOT / args.split_file
    if split_path.exists():
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        splits = SplitIndices(
            train=[int(value) for value in payload["train"]],
            val=[int(value) for value in payload["val"]],
            test=[int(value) for value in payload["test"]],
        )
        expected = {
            "train": args.train_per_class * 2,
            "val": args.val_per_class * 2,
            "test": args.test_per_class * 2,
        }
        actual = {"train": len(splits.train), "val": len(splits.val), "test": len(splits.test)}
        if actual == expected:
            return splits
        print(f"Ignoring stale split file with sizes {actual}; expected {expected}", flush=True)
    splits = make_fixed_splits(
        correct_indices,
        wrong_indices,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps({"train": splits.train, "val": splits.val, "test": splits.test}, indent=2), encoding="utf-8")
    return splits


def cache_path(args: argparse.Namespace) -> Path:
    return PROJECT_ROOT / args.cache_dir / f"top_global_{args.edges_per_layer}_seed{args.seed}_graphs.pt"


def extract_sparse_graph_cache(
    args: argparse.Namespace,
    dataset: Dataset,
    indices: Sequence[int],
    correct_lookup: set[int],
    processor,
    vit_model,
    device: torch.device,
) -> SparseGraphCache:
    path = cache_path(args)
    if path.exists():
        payload = torch.load(path, map_location="cpu")
        return SparseGraphCache(payload["rows"], layer_count=int(payload["layer_count"]))

    subset = Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=args.vit_batch_size, shuffle=False, num_workers=0, collate_fn=collate_images)
    rows: List[Dict[str, object]] = []
    layer_count = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc="extract top-global graphs")):
        images, labels = batch
        pixel_values = preprocess_batch(processor, images, device)
        with torch.no_grad():
            outputs = vit_model(pixel_values, output_attentions=True)
            logits = outputs.logits.detach().cpu()
            probabilities = torch.softmax(logits, dim=-1)
            confidence, pred_label = probabilities.max(dim=-1)
            attentions = torch.stack([attention.detach().cpu() for attention in outputs.attentions], dim=1)
        layer_count = int(attentions.shape[1])
        for item_idx in range(logits.shape[0]):
            global_index = int(indices[batch_idx * args.vit_batch_size + item_idx])
            true_label = int(labels[item_idx])
            prediction = int(pred_label[item_idx])
            layers = [
                build_top_global_graph(attentions[item_idx, layer_idx], args.edges_per_layer, layer_idx, layer_count)
                for layer_idx in range(layer_count)
            ]
            vit_correct = prediction == true_label
            if vit_correct != (global_index in correct_lookup):
                raise RuntimeError(f"Cached scan correctness mismatch at CIFAR index {global_index}.")
            rows.append(
                {
                    "image_id": global_index,
                    "true_label": true_label,
                    "pred_label": prediction,
                    "vit_correct": float(vit_correct),
                    "y_err": float(not vit_correct),
                    "confidence": float(confidence[item_idx]),
                    "layers": layers,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": rows, "layer_count": layer_count}, path)
    return SparseGraphCache(rows, layer_count=layer_count)


class GNNEncoder(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * layers
        self.layers = nn.ModuleList(
            [TransformerConv(dims[idx], dims[idx + 1], heads=2, concat=False, edge_dim=edge_dim) for idx in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, edge_index, edge_attr)
            x = self.dropout(torch.relu(norm(x)))
        return x


class SingleLayerModel(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.encoder = GNNEncoder(in_dim, edge_dim, hidden_dim, layers, dropout)
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Data) -> Tuple[Tensor, Tensor]:
        x = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        embedding = global_mean_pool(x, batch.batch)
        return self.decoder(embedding).view(-1), embedding


class ReadoutModel(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, layers: int, dropout: float, readout: str):
        super().__init__()
        if readout not in {"mean", "cls", "cls_mean", "cls_mean_max", "cls_gated"}:
            raise ValueError(f"Unknown readout: {readout}")
        self.readout = readout
        self.encoder = GNNEncoder(in_dim, edge_dim, hidden_dim, layers, dropout)
        if readout in {"mean", "cls"}:
            decoder_dim = hidden_dim
        elif readout in {"cls_mean", "cls_gated"}:
            decoder_dim = hidden_dim * 2
        else:
            decoder_dim = hidden_dim * 3
        if readout == "cls_gated":
            self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(decoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Data) -> Tuple[Tensor, Tensor]:
        raw_x = batch.x
        x = self.encoder(raw_x, batch.edge_index, batch.edge_attr)
        graph_count = int(batch.batch.max().item()) + 1 if batch.batch.numel() else 0
        cls_mask = raw_x[:, 2] > 0.5
        cls_nodes = x[cls_mask]
        if cls_nodes.shape[0] != graph_count:
            raise RuntimeError(f"Expected one CLS node per graph, got {cls_nodes.shape[0]} for {graph_count} graphs.")
        mean_nodes = global_mean_pool(x, batch.batch)
        if self.readout == "mean":
            embedding = mean_nodes
        elif self.readout == "cls":
            embedding = cls_nodes
        elif self.readout == "cls_mean":
            embedding = torch.cat([cls_nodes, mean_nodes], dim=1)
        elif self.readout == "cls_mean_max":
            max_nodes = global_max_pool(x, batch.batch)
            embedding = torch.cat([cls_nodes, mean_nodes, max_nodes], dim=1)
        else:
            scores = self.gate(x).view(-1)
            weights = softmax(scores, batch.batch)
            gated = global_add_pool(x * weights.unsqueeze(1), batch.batch)
            embedding = torch.cat([cls_nodes, gated], dim=1)
        return self.decoder(embedding).view(-1), embedding


class SequenceConcatModel(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, layers: int, dropout: float, layer_count: int):
        super().__init__()
        self.layer_count = layer_count
        self.encoder = GNNEncoder(in_dim, edge_dim, hidden_dim, layers, dropout)
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * layer_count, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Data) -> Tuple[Tensor, Tensor]:
        x = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        graph_count = int(batch.batch.max().item()) + 1 if batch.batch.numel() else 0
        layer_graph_id = batch.batch * self.layer_count + batch.layer_id
        layer_embeddings = global_mean_pool(x, layer_graph_id, size=graph_count * self.layer_count)
        ordered = layer_embeddings.view(graph_count, self.layer_count, -1).reshape(graph_count, -1)
        return self.decoder(ordered).view(-1), ordered


class FinalFourGatedFusionModel(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, layers: int, dropout: float, layer_count: int):
        super().__init__()
        self.layer_count = layer_count
        self.hidden_dim = hidden_dim
        self.readout_dim = hidden_dim * 2
        self.encoder = GNNEncoder(in_dim, edge_dim, hidden_dim, layers, dropout)
        self.node_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.layer_gate = nn.Sequential(nn.Linear(self.readout_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.readout_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Data) -> Tuple[Tensor, Tensor]:
        raw_x = batch.x
        x = self.encoder(raw_x, batch.edge_index, batch.edge_attr)
        graph_count = int(batch.batch.max().item()) + 1 if batch.batch.numel() else 0
        layer_graph_id = batch.batch * self.layer_count + batch.layer_id
        cls_mask = raw_x[:, 2] > 0.5
        cls_nodes = x[cls_mask]
        if cls_nodes.shape[0] != graph_count * self.layer_count:
            raise RuntimeError(
                f"Expected one CLS per layer graph, got {cls_nodes.shape[0]} for {graph_count * self.layer_count}."
            )
        node_scores = self.node_gate(x).view(-1)
        node_weights = softmax(node_scores, layer_graph_id)
        gated_nodes = global_add_pool(
            x * node_weights.unsqueeze(1),
            layer_graph_id,
            size=graph_count * self.layer_count,
        )
        layer_readouts = torch.cat([cls_nodes, gated_nodes], dim=1).view(graph_count, self.layer_count, self.readout_dim)
        layer_scores = self.layer_gate(layer_readouts).squeeze(-1)
        layer_weights = torch.softmax(layer_scores, dim=1)
        fused = (layer_readouts * layer_weights.unsqueeze(-1)).sum(dim=1)
        final_layer = layer_readouts[:, -1, :]
        embedding = torch.cat([fused, final_layer], dim=1)
        return self.decoder(embedding).view(-1), embedding


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_optimizer_coverage(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    grouped: List[int] = []
    for group in optimizer.param_groups:
        grouped.extend(id(parameter) for parameter in group["params"])
    if len(grouped) != len(set(grouped)):
        raise RuntimeError("A trainable parameter appears in more than one optimizer group.")
    if trainable != set(grouped):
        raise RuntimeError("Optimizer parameter groups do not exactly cover trainable model parameters.")


def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().cpu())
    return math.sqrt(total)


def safe_auroc(labels: np.ndarray, probs: np.ndarray) -> Optional[float]:
    if len(np.unique(labels)) <= 1:
        return None
    return float(roc_auc_score(labels, probs))


def safe_auprc(labels: np.ndarray, probs: np.ndarray) -> Optional[float]:
    if len(np.unique(labels)) <= 1:
        return None
    return float(average_precision_score(labels, probs))


def evaluate(model: nn.Module, loader: PyGDataLoader, device: torch.device, criterion: nn.Module) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total = 0
    probs: List[float] = []
    logits_out: List[float] = []
    labels_out: List[float] = []
    embeddings: List[Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, embedding = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            probabilities = torch.sigmoid(logits)
            total_loss += float(loss.item()) * batch.y.numel()
            total += batch.y.numel()
            probs.extend(probabilities.detach().cpu().numpy().tolist())
            logits_out.extend(logits.detach().cpu().numpy().tolist())
            labels_out.extend(batch.y.view(-1).detach().cpu().numpy().tolist())
            embeddings.append(embedding.detach().cpu())
    probabilities_np = np.asarray(probs)
    labels_np = np.asarray(labels_out)
    logits_np = np.asarray(logits_out)
    embedding_tensor = torch.cat(embeddings, dim=0) if embeddings else torch.empty((0, 1))
    predictions = probabilities_np >= 0.5
    return {
        "loss": float(total_loss / max(total, 1)),
        "accuracy": float(accuracy_score(labels_np, predictions)) if labels_np.size else 0.0,
        "auroc": safe_auroc(labels_np, probabilities_np),
        "auprc": safe_auprc(labels_np, probabilities_np),
        "prob_mean": float(probabilities_np.mean()) if probabilities_np.size else 0.0,
        "prob_std": float(probabilities_np.std()) if probabilities_np.size else 0.0,
        "logit_min": float(logits_np.min()) if logits_np.size else 0.0,
        "logit_max": float(logits_np.max()) if logits_np.size else 0.0,
        "logit_mean": float(logits_np.mean()) if logits_np.size else 0.0,
        "logit_std": float(logits_np.std()) if logits_np.size else 0.0,
        "embedding_mean": float(embedding_tensor.mean().item()) if embedding_tensor.numel() else 0.0,
        "embedding_var": float(embedding_tensor.var(unbiased=False).item()) if embedding_tensor.numel() else 0.0,
        "positives": int(labels_np.sum()) if labels_np.size else 0,
        "total": int(labels_np.shape[0]),
    }


def train_model(
    name: str,
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
    epochs: Optional[int] = None,
    dropout_free_overfit: bool = False,
) -> Dict[str, object]:
    del dropout_free_overfit
    epochs = args.epochs if epochs is None else epochs
    train_loader = PyGDataLoader(train_dataset, batch_size=args.gnn_batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=args.gnn_batch_size, shuffle=False) if len(val_dataset) else None
    test_loader = PyGDataLoader(test_dataset, batch_size=args.gnn_batch_size, shuffle=False)
    positive_count = 0
    for idx in range(len(train_dataset)):
        positive_count += int(train_dataset[idx].y.item() > 0.5)
    negative_count = max(len(train_dataset) - positive_count, 1)
    pos_weight = torch.tensor([negative_count / max(positive_count, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    assert_optimizer_coverage(model, optimizer)

    best_state: Optional[Dict[str, Tensor]] = None
    best_val_auroc = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    epoch_rows: List[Dict[str, object]] = []
    start = time.time()
    print(f"Starting {name}: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}, epochs={epochs}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        first_grad = 0.0
        decoder_grad = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            first_grad = grad_norm(model.encoder.layers[0])
            decoder_grad = grad_norm(model.decoder)
            optimizer.step()
            running_loss += float(loss.item()) * batch.y.numel()
            seen += batch.y.numel()
        train_eval = evaluate(model, train_loader, device, criterion)
        val_eval = evaluate(model, val_loader, device, criterion) if val_loader is not None else None
        val_auroc = val_eval["auroc"] if val_eval is not None and val_eval["auroc"] is not None else -float("inf")
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(running_loss / max(seen, 1)),
            "train_eval_loss": train_eval["loss"],
            "train_auroc": train_eval["auroc"],
            "val_loss": None if val_eval is None else val_eval["loss"],
            "val_auroc": None if val_eval is None else val_eval["auroc"],
            "val_auprc": None if val_eval is None else val_eval["auprc"],
            "val_prob_mean": None if val_eval is None else val_eval["prob_mean"],
            "val_prob_std": None if val_eval is None else val_eval["prob_std"],
            "val_logit_min": None if val_eval is None else val_eval["logit_min"],
            "val_logit_max": None if val_eval is None else val_eval["logit_max"],
            "val_logit_mean": None if val_eval is None else val_eval["logit_mean"],
            "val_logit_std": None if val_eval is None else val_eval["logit_std"],
            "val_embedding_mean": None if val_eval is None else val_eval["embedding_mean"],
            "val_embedding_var": None if val_eval is None else val_eval["embedding_var"],
            "first_gnn_grad_norm": first_grad,
            "decoder_grad_norm": decoder_grad,
        }
        epoch_rows.append(epoch_row)
        improved = val_loader is not None and val_auroc > best_val_auroc + args.min_delta
        if improved:
            best_val_auroc = float(val_auroc)
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if val_loader is not None and args.early_stopping and stale_epochs >= args.patience:
            print(
                f"  {name} epoch {epoch}: early stop, val_auroc={val_eval['auroc']}, val_loss={val_eval['loss']:.4f}",
                flush=True,
            )
            break
        if epoch == 1 or epoch % args.log_every == 0 or epoch == epochs:
            if val_eval is None:
                progress_tail = "no_val=true"
            else:
                progress_tail = f"val_auroc={val_eval['auroc']}, val_prob_std={val_eval['prob_std']:.4f}"
            print(
                f"  {name} epoch {epoch}: train_loss={running_loss / max(seen, 1):.4f}, "
                f"train_auroc={train_eval['auroc']}, {progress_tail}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_train = evaluate(model, train_loader, device, criterion)
    final_val = evaluate(model, val_loader, device, criterion) if val_loader is not None else {}
    final_test = evaluate(model, test_loader, device, criterion)
    return {
        "name": name,
        "best_epoch": best_epoch,
        "epochs_ran": len(epoch_rows),
        "parameter_count": parameter_count(model),
        "elapsed_seconds": time.time() - start,
        "train": final_train,
        "val": final_val,
        "test": final_test,
        "epoch_log": epoch_rows,
    }


def make_model(dataset: Dataset, args: argparse.Namespace, device: torch.device, sequence_layer_count: int) -> nn.Module:
    first = dataset[0]
    in_dim = int(first.x.shape[1])
    edge_dim = int(first.edge_attr.shape[1])
    if sequence_layer_count == 1 and not args.force_sequence_model:
        model = SingleLayerModel(in_dim, edge_dim, args.hidden_dim, args.gnn_layers, args.dropout)
    else:
        model = SequenceConcatModel(in_dim, edge_dim, args.hidden_dim, args.gnn_layers, args.dropout, sequence_layer_count)
    return model.to(device)


def balanced_subset(rows: Sequence[Dict[str, object]], correct_count: int, wrong_count: int, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    correct = [row for row in rows if float(row["y_err"]) < 0.5]
    wrong = [row for row in rows if float(row["y_err"]) > 0.5]
    rng.shuffle(correct)
    rng.shuffle(wrong)
    subset = correct[:correct_count] + wrong[:wrong_count]
    rng.shuffle(subset)
    return subset


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_result(result: Dict[str, object], prefix: str = "") -> Dict[str, object]:
    flat: Dict[str, object] = {}
    for key, value in result.items():
        if key == "epoch_log":
            continue
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                flat[f"{prefix}{key}_{inner_key}"] = inner_value
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def run_stage1(cache: SparseGraphCache, splits: SplitIndices, args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    output_dir = PROJECT_ROOT / args.output_dir
    train_rows = cache.select(splits.train)
    val_rows = cache.select(splits.val)
    test_rows = cache.select(splits.test)
    tiny_rows = balanced_subset(train_rows, 16, 16, args.seed)
    print("Stage 1: metadata check", flush=True)

    metadata_report = []
    full_sequence_dataset = LayerGraphDataset(train_rows[:2], list(range(cache.layer_count)), cache.layer_count)
    for item_idx in range(len(full_sequence_dataset)):
        data = full_sequence_dataset[item_idx]
        counts = torch.bincount(data.layer_id, minlength=cache.layer_count).tolist()
        metadata_report.append(
            {
                "item": item_idx,
                "image_id": int(data.image_id.item()),
                "target": float(data.y.item()),
                "layer_node_counts": counts,
                "edge_count": int(data.edge_index.shape[1]),
            }
        )
        if any(count <= 0 for count in counts):
            raise RuntimeError("At least one sequence layer has no nodes in the metadata check.")
        if int(data.edge_index.shape[1]) != cache.layer_count * args.edges_per_layer:
            raise RuntimeError("Unexpected edge count in sequence metadata check.")

    original_dropout = args.dropout
    original_weight_decay = args.weight_decay
    original_epochs = args.epochs
    original_early = args.early_stopping
    original_batch = args.gnn_batch_size
    args.dropout = 0.0
    args.weight_decay = 0.0
    args.epochs = args.overfit_epochs
    args.early_stopping = False
    args.gnn_batch_size = min(args.gnn_batch_size, 16)
    args.force_sequence_model = True
    print("Stage 1: tiny sequence overfit", flush=True)
    tiny_dataset = LayerGraphDataset(tiny_rows, list(range(cache.layer_count)), cache.layer_count)
    tiny_model = make_model(tiny_dataset, args, device, sequence_layer_count=cache.layer_count)
    tiny_result = train_model("stage1_tiny_overfit_sequence", tiny_model, tiny_dataset, tiny_dataset, tiny_dataset, args, device)
    args.dropout = original_dropout
    args.weight_decay = original_weight_decay
    args.epochs = original_epochs
    args.early_stopping = original_early
    args.gnn_batch_size = original_batch

    args.force_sequence_model = False
    print("Stage 1: final-layer single-path reference", flush=True)
    final_layer_dataset_train = LayerGraphDataset(train_rows, [cache.layer_count - 1], cache.layer_count)
    final_layer_dataset_val = LayerGraphDataset(val_rows, [cache.layer_count - 1], cache.layer_count)
    final_layer_dataset_test = LayerGraphDataset(test_rows, [cache.layer_count - 1], cache.layer_count)
    final_model = make_model(final_layer_dataset_train, args, device, sequence_layer_count=1)
    final_result = train_model(
        "stage1_final_layer_single_path",
        final_model,
        final_layer_dataset_train,
        final_layer_dataset_val,
        final_layer_dataset_test,
        args,
        device,
    )

    args.force_sequence_model = True
    print("Stage 1: final-layer sequence length one", flush=True)
    sequence_one_train = LayerGraphDataset(train_rows, [cache.layer_count - 1], cache.layer_count)
    sequence_one_val = LayerGraphDataset(val_rows, [cache.layer_count - 1], cache.layer_count)
    sequence_one_test = LayerGraphDataset(test_rows, [cache.layer_count - 1], cache.layer_count)
    sequence_one_model = make_model(sequence_one_train, args, device, sequence_layer_count=1)
    sequence_one_result = train_model(
        "stage1_final_layer_sequence_len1",
        sequence_one_model,
        sequence_one_train,
        sequence_one_val,
        sequence_one_test,
        args,
        device,
    )

    print("Stage 1: duplicate-final-layer sequence", flush=True)
    duplicate_train = LayerGraphDataset(train_rows, list(range(cache.layer_count)), cache.layer_count, duplicate_final=True)
    duplicate_val = LayerGraphDataset(val_rows, list(range(cache.layer_count)), cache.layer_count, duplicate_final=True)
    duplicate_test = LayerGraphDataset(test_rows, list(range(cache.layer_count)), cache.layer_count, duplicate_final=True)
    duplicate_model = make_model(duplicate_train, args, device, sequence_layer_count=cache.layer_count)
    duplicate_result = train_model(
        "stage1_duplicate_final_sequence",
        duplicate_model,
        duplicate_train,
        duplicate_val,
        duplicate_test,
        args,
        device,
    )
    args.force_sequence_model = False

    payload = {
        "edge_mode": "top_global",
        "edges_per_layer": args.edges_per_layer,
        "metadata_report": metadata_report,
        "tiny_overfit": tiny_result,
        "final_layer_single_path": final_result,
        "final_layer_sequence_len1": sequence_one_result,
        "duplicate_final_sequence": duplicate_result,
    }
    write_json(output_dir / "stage1_sanity_results.json", payload)
    lines = [
        "# Stage 1 Sequence Debug Report",
        "",
        f"Edge mode: top_global, {args.edges_per_layer} edges per layer.",
        "",
        "## Metadata Check",
        json.dumps(metadata_report, indent=2),
        "",
        "## Results",
        "| Run | Train AUROC | Train Loss | Val AUROC | Test AUROC | Test AUPRC | Test Acc | Epochs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in [tiny_result, final_result, sequence_one_result, duplicate_result]:
        val_auroc = result["val"].get("auroc") if isinstance(result["val"], dict) else None
        lines.append(
            f"| {result['name']} | {result['train']['auroc']} | {result['train']['loss']:.4f} | "
            f"{val_auroc} | {result['test']['auroc']} | {result['test']['auprc']} | "
            f"{result['test']['accuracy']:.4f} | {result['epochs_ran']} |"
        )
    (output_dir / "stage1_sequence_debug_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def run_stage2(cache: SparseGraphCache, splits: SplitIndices, args: argparse.Namespace, device: torch.device) -> List[Dict[str, object]]:
    output_dir = PROJECT_ROOT / args.output_dir
    train_rows = cache.select(splits.train)
    val_rows = cache.select(splits.val)
    test_rows = cache.select(splits.test)
    rows: List[Dict[str, object]] = []
    for layer_idx in range(cache.layer_count):
        args.force_sequence_model = False
        train_dataset = LayerGraphDataset(train_rows, [layer_idx], cache.layer_count)
        val_dataset = LayerGraphDataset(val_rows, [layer_idx], cache.layer_count)
        test_dataset = LayerGraphDataset(test_rows, [layer_idx], cache.layer_count)
        model = make_model(train_dataset, args, device, sequence_layer_count=1)
        result = train_model(f"stage2_layer_{layer_idx}", model, train_dataset, val_dataset, test_dataset, args, device)
        row = flatten_result(result)
        row.update(
            {
                "layer_index_zero_based": layer_idx,
                "layer_index_one_based": layer_idx + 1,
                "mean_edges_per_graph": args.edges_per_layer,
            }
        )
        rows.append(row)
        write_json(output_dir / f"stage2_layer_{layer_idx}_result.json", result)
        write_csv(output_dir / "stage2_layer_sweep_partial.csv", rows)
    write_csv(output_dir / "stage2_layer_sweep.csv", rows)
    best = max(rows, key=lambda row: -1.0 if row.get("val_auroc") is None else float(row.get("val_auroc")))
    report = [
        "# Stage 2 Layer Sweep Report",
        "",
        f"Edge mode: top_global, {args.edges_per_layer} edges per layer.",
        f"Best validation layer: {best['layer_index_one_based']} one-based.",
        "",
        "| Layer | Val AUROC | Test AUROC | Test AUPRC | Test Acc | Best Epoch | Runtime s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['layer_index_one_based']} | {row.get('val_auroc')} | {row.get('test_auroc')} | "
            f"{row.get('test_auprc')} | {row.get('test_accuracy')} | {row.get('best_epoch')} | "
            f"{float(row.get('elapsed_seconds', 0.0)):.1f} |"
        )
    (output_dir / "stage2_layer_sweep_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return rows


def run_stage3(cache: SparseGraphCache, splits: SplitIndices, args: argparse.Namespace, device: torch.device) -> List[Dict[str, object]]:
    output_dir = PROJECT_ROOT / args.output_dir
    train_rows = cache.select(splits.train)
    test_rows = cache.select(splits.test)
    layer_idx = cache.layer_count - 1
    train_dataset = LayerGraphDataset(train_rows, [layer_idx], cache.layer_count)
    test_dataset = LayerGraphDataset(test_rows, [layer_idx], cache.layer_count)
    rows: List[Dict[str, object]] = []
    for readout in ["mean", "cls", "cls_mean", "cls_mean_max", "cls_gated"]:
        first = train_dataset[0]
        model = ReadoutModel(
            in_dim=int(first.x.shape[1]),
            edge_dim=int(first.edge_attr.shape[1]),
            hidden_dim=args.hidden_dim,
            layers=args.gnn_layers,
            dropout=args.dropout,
            readout=readout,
        ).to(device)
        result = train_model(
            f"stage3_layer_{layer_idx}_readout_{readout}",
            model,
            train_dataset,
            [],
            test_dataset,
            args,
            device,
        )
        row = flatten_result(result)
        row.update({"readout": readout, "layer_index_zero_based": layer_idx, "layer_index_one_based": layer_idx + 1})
        rows.append(row)
        write_json(output_dir / f"stage3_readout_{readout}_result.json", result)
        write_csv(output_dir / "stage3_pooling_comparison_partial.csv", rows)
    write_csv(output_dir / "stage3_pooling_comparison.csv", rows)
    best = max(rows, key=lambda row: -1.0 if row.get("test_auroc") is None else float(row.get("test_auroc")))
    report = [
        "# Stage 3 Pooling Comparison Report",
        "",
        f"Layer: {layer_idx + 1} one-based. Edge mode: top_global, {args.edges_per_layer} edges.",
        f"Best readout by test AUROC: {best['readout']}.",
        "",
        "| Readout | Test AUROC | Test AUPRC | Test Acc | Train AUROC | Runtime s | Params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['readout']} | {row.get('test_auroc')} | {row.get('test_auprc')} | "
            f"{row.get('test_accuracy')} | {row.get('train_auroc')} | "
            f"{float(row.get('elapsed_seconds', 0.0)):.1f} | {row.get('parameter_count')} |"
        )
    (output_dir / "stage3_pooling_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return rows


def run_stage4(cache: SparseGraphCache, splits: SplitIndices, args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    output_dir = PROJECT_ROOT / args.output_dir
    train_rows = cache.select(splits.train)
    test_rows = cache.select(splits.test)
    selected_layers = list(range(cache.layer_count - args.fusion_layer_count, cache.layer_count))
    train_dataset = LayerGraphDataset(train_rows, selected_layers, cache.layer_count)
    test_dataset = LayerGraphDataset(test_rows, selected_layers, cache.layer_count)
    first = train_dataset[0]
    model = FinalFourGatedFusionModel(
        in_dim=int(first.x.shape[1]),
        edge_dim=int(first.edge_attr.shape[1]),
        hidden_dim=args.hidden_dim,
        layers=args.gnn_layers,
        dropout=args.dropout,
        layer_count=len(selected_layers),
    ).to(device)
    result = train_model(
        "stage4_final_four_cls_gated_layer_fusion",
        model,
        train_dataset,
        [],
        test_dataset,
        args,
        device,
    )
    payload = {
        "selected_layers_zero_based": selected_layers,
        "selected_layers_one_based": [layer + 1 for layer in selected_layers],
        "readout": "per-layer CLS + gated node pooling",
        "fusion": "learned layer gate + final-layer skip",
        "result": result,
    }
    write_json(output_dir / f"stage4_final_{args.fusion_layer_count}_gated_fusion_results.json", payload)
    row = flatten_result(result)
    row.update(
        {
            "selected_layers_one_based": "-".join(str(layer + 1) for layer in selected_layers),
            "readout": "cls_gated",
            "fusion": "learned_layer_gate_final_skip",
        }
    )
    write_csv(output_dir / f"stage4_final_{args.fusion_layer_count}_fusion_results.csv", [row])
    report = [
        f"# Stage 4 Final-{args.fusion_layer_count} Gated Fusion Report",
        "",
        f"Layers: {[layer + 1 for layer in selected_layers]} one-based.",
        "Readout: per-layer CLS + gated node pooling.",
        "Fusion: learned layer gate with final-layer skip.",
        "",
        "| Test AUROC | Test AUPRC | Test Acc | Train AUROC | Train Loss | Runtime s | Params |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {result['test']['auroc']} | {result['test']['auprc']} | {result['test']['accuracy']} | "
        f"{result['train']['auroc']} | {result['train']['loss']:.4f} | {result['elapsed_seconds']:.1f} | "
        f"{result['parameter_count']} |",
    ]
    (output_dir / f"stage4_final_{args.fusion_layer_count}_fusion_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def collect_predictions(model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int) -> Dict[str, np.ndarray]:
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    image_ids: List[int] = []
    labels: List[float] = []
    confidences: List[float] = []
    logits_out: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            logits_out.extend(logits.detach().cpu().numpy().tolist())
            labels.extend(batch.y.view(-1).detach().cpu().numpy().tolist())
            image_ids.extend(batch.image_id.view(-1).detach().cpu().numpy().tolist())
            confidences.extend(batch.confidence.view(-1).detach().cpu().numpy().tolist())
    return {
        "image_id": np.asarray(image_ids),
        "y_err": np.asarray(labels),
        "confidence": np.asarray(confidences),
        "graph_logit": np.asarray(logits_out),
    }


def run_stage5(cache: SparseGraphCache, splits: SplitIndices, args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    """Test whether the attention-graph score adds signal beyond ViT max-softmax confidence."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    output_dir = PROJECT_ROOT / args.output_dir
    train_rows = cache.select(splits.train)
    test_rows = cache.select(splits.test)
    layer_idx = cache.layer_count - 1
    train_dataset = LayerGraphDataset(train_rows, [layer_idx], cache.layer_count)
    test_dataset = LayerGraphDataset(test_rows, [layer_idx], cache.layer_count)

    first = train_dataset[0]
    model = ReadoutModel(
        in_dim=int(first.x.shape[1]),
        edge_dim=int(first.edge_attr.shape[1]),
        hidden_dim=args.hidden_dim,
        layers=args.gnn_layers,
        dropout=args.dropout,
        readout="cls_gated",
    ).to(device)
    result = train_model("stage5_layer12_cls_gated", model, train_dataset, [], test_dataset, args, device)

    train_pred = collect_predictions(model, train_dataset, device, args.gnn_batch_size)
    test_pred = collect_predictions(model, test_dataset, device, args.gnn_batch_size)

    def combiner_features(pred: Dict[str, np.ndarray]) -> np.ndarray:
        # Raw MSP uncertainty is tiny and skewed; log-odds is the scale a logistic model is linear in,
        # and it is monotone in confidence so MSP-alone ranking (AUROC) is preserved.
        conf = np.clip(pred["confidence"], 1e-6, 1.0 - 1e-6)
        msp_logit = np.log((1.0 - conf) / conf)
        return np.stack([pred["graph_logit"], msp_logit], axis=1)

    train_features = combiner_features(train_pred)
    test_features = combiner_features(test_pred)

    # Fit the combiner on train only so the test comparison stays honest.
    # Standardize first: without it the default L2 penalty suppresses the small-scale MSP feature,
    # which made the combiner score below MSP alone.
    combiner = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    combiner.fit(train_features, train_pred["y_err"])
    combined_score = combiner.predict_proba(test_features)[:, 1]

    y = test_pred["y_err"]
    graph_score = test_pred["graph_logit"]
    msp_score = 1.0 - test_pred["confidence"]
    conf = test_pred["confidence"]

    overall = {
        "msp_auroc": safe_auroc(y, msp_score),
        "msp_auprc": safe_auprc(y, msp_score),
        "graph_auroc": safe_auroc(y, graph_score),
        "graph_auprc": safe_auprc(y, graph_score),
        "combined_auroc": safe_auroc(y, combined_score),
        "combined_auprc": safe_auprc(y, combined_score),
        # Coefficients are on the standardized feature scale, so they are directly comparable.
        "combiner_coef_graph": float(combiner[-1].coef_[0][0]),
        "combiner_coef_msp": float(combiner[-1].coef_[0][1]),
    }

    subsets: List[Dict[str, object]] = []
    for threshold in args.confidence_thresholds:
        mask = conf >= threshold
        n_wrong = int(y[mask].sum())
        n_correct = int((y[mask] == 0).sum())
        usable = bool(n_wrong and n_correct)
        subsets.append(
            {
                "confidence_threshold": threshold,
                "n_images": int(mask.sum()),
                "n_wrong": n_wrong,
                "n_correct": n_correct,
                "msp_auroc": safe_auroc(y[mask], msp_score[mask]) if usable else None,
                "graph_auroc": safe_auroc(y[mask], graph_score[mask]) if usable else None,
                "combined_auroc": safe_auroc(y[mask], combined_score[mask]) if usable else None,
            }
        )

    correlation = float(np.corrcoef(graph_score, msp_score)[0, 1])
    payload = {
        "layer_one_based": layer_idx + 1,
        "readout": "cls_gated",
        "overall": overall,
        "high_confidence_subsets": subsets,
        "graph_msp_score_correlation": correlation,
        "training_result": result,
    }
    write_json(output_dir / "stage5_complementarity_results.json", payload)

    per_example = [
        {
            "image_id": int(test_pred["image_id"][i]),
            "y_err": float(y[i]),
            "vit_confidence": float(conf[i]),
            "graph_logit": float(graph_score[i]),
            "msp_uncertainty": float(msp_score[i]),
            "combined_score": float(combined_score[i]),
        }
        for i in range(len(y))
    ]
    write_csv(output_dir / "stage5_test_predictions.csv", per_example)
    write_csv(output_dir / "stage5_confidence_subsets.csv", subsets)

    def fmt(value: object) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"

    report = [
        "# Stage 5 Complementarity Report",
        "",
        f"Layer {layer_idx + 1}, readout cls_gated, top_global {args.edges_per_layer} edges.",
        "Combiner: logistic regression on standardized [graph_logit, logit(msp_uncertainty)], fit on train only.",
        "",
        "## Overall test performance",
        "| Detector | AUROC | AUPRC |",
        "|---|---:|---:|",
        f"| MSP only | {fmt(overall['msp_auroc'])} | {fmt(overall['msp_auprc'])} |",
        f"| Graph only | {fmt(overall['graph_auroc'])} | {fmt(overall['graph_auprc'])} |",
        f"| MSP + graph | {fmt(overall['combined_auroc'])} | {fmt(overall['combined_auprc'])} |",
        "",
        f"Correlation between graph score and MSP uncertainty: {correlation:.4f}",
        "",
        "## Restricted to high-confidence images (where MSP is weakest)",
        "| Conf >= | Images | Wrong | Correct | MSP AUROC | Graph AUROC | Combined AUROC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subset in subsets:
        report.append(
            f"| {subset['confidence_threshold']} | {subset['n_images']} | {subset['n_wrong']} | "
            f"{subset['n_correct']} | {fmt(subset['msp_auroc'])} | {fmt(subset['graph_auroc'])} | "
            f"{fmt(subset['combined_auroc'])} |"
        )
    (output_dir / "stage5_complementarity_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def prepare(args: argparse.Namespace):
    seed_everything(args.seed)
    device = choose_device()
    print(f"Using device: {device}", flush=True)

    scan_cache = PROJECT_ROOT / args.scan_cache
    # The ViT is only needed to build caches; skip loading it (and hitting the network) when both already exist.
    offline = cache_path(args).exists() and scan_cache.exists()
    if offline:
        print("Using cached graphs and scan; skipping ViT load.", flush=True)
        processor = None
        vit_model = None
        dataset = None
    else:
        model_id = pick_model(args.model_id)
        print(f"Using model: {model_id}", flush=True)
        processor, vit_model = load_vit(model_id, device)
        dataset = load_local_hf_parquet_cifar100(args.split, 0, args.seed)

    if scan_cache.exists():
        scan_summary, correct_indices, wrong_indices = read_scan_cache(scan_cache)
    else:
        scan_summary, correct_indices, wrong_indices = scan_vit_correctness(
            dataset, list(range(len(dataset))), processor, vit_model, device, args.scan_batch_size
        )
        write_scan_cache(scan_cache, scan_summary, correct_indices, wrong_indices)
    splits = load_or_create_splits(args, correct_indices, wrong_indices)
    print(f"Split sizes: train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}", flush=True)
    needed_indices = sorted(set(splits.train + splits.val + splits.test))
    print(f"Needed sparse graph rows: {len(needed_indices)}", flush=True)
    cache = extract_sparse_graph_cache(args, dataset, needed_indices, set(correct_indices), processor, vit_model, device)
    print(f"Loaded sparse graph cache: rows={len(cache.rows)}, layers={cache.layer_count}", flush=True)
    return cache, splits, scan_summary, device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["stage1", "stage2", "stage3", "stage4", "stage5", "stage1-stage2"],
        default="stage1",
    )
    parser.add_argument("--model-id", default="edumunozsala/vit_base-224-in21k-ft-cifar100")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scan-cache", default="reports_vit_scan_test/vit_scan_with_indices.json")
    parser.add_argument("--split-file", default="reports_lightweight_top100/experiment_splits_train1500_test200_seed7.json")
    parser.add_argument("--cache-dir", default="reports_lightweight_top100/cache")
    parser.add_argument("--output-dir", default="reports_lightweight_top100")
    parser.add_argument("--train-per-class", type=int, default=750)
    parser.add_argument("--val-per-class", type=int, default=0)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--edges-per-layer", type=int, default=100)
    parser.add_argument("--scan-batch-size", type=int, default=8)
    parser.add_argument("--vit-batch-size", type=int, default=2)
    parser.add_argument("--gnn-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--overfit-epochs", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--fusion-layer-count", type=int, default=4)
    parser.add_argument("--confidence-thresholds", type=float, nargs="*", default=[0.0, 0.8, 0.9, 0.95, 0.99])
    parser.set_defaults(force_sequence_model=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    cache, splits, scan_summary, device = prepare(args)
    output_dir = PROJECT_ROOT / args.output_dir
    config = vars(args).copy()
    config.update({"device": str(device), "scan_summary": scan_summary, "layer_count": cache.layer_count})
    write_json(output_dir / "experiment_config.json", config)
    if args.stage in {"stage1", "stage1-stage2"}:
        run_stage1(cache, splits, args, device)
    if args.stage in {"stage2", "stage1-stage2"}:
        run_stage2(cache, splits, args, device)
    if args.stage == "stage3":
        run_stage3(cache, splits, args, device)
    if args.stage == "stage4":
        run_stage4(cache, splits, args, device)
    if args.stage == "stage5":
        run_stage5(cache, splits, args, device)
    print(f"Finished {args.stage} in {time.time() - start:.1f}s. Outputs: {output_dir}")


if __name__ == "__main__":
    main()