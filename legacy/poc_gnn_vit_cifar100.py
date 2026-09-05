#!/usr/bin/env python3
"""POC: frozen ViT attention graphs for CIFAR-100 failure prediction.

Everything is written relative to the current working directory. The script:
1. loads a CIFAR-100-finetuned ViT from HuggingFace without training it;
2. computes correctness labels, attentions, and hidden states on a small subset;
3. builds sparse per-image attention graphs from the last ViT layer;
4. trains PyTorch Geometric GNN detectors above the frozen ViT;
5. writes metrics and a markdown report under reports/.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Keep all common caches inside the project folder. These must be set before HF imports.
PROJECT_ROOT = Path.cwd().resolve()
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "datasets"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))

import numpy as np
import torch
from huggingface_hub import HfApi
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATv2Conv, GCNConv, NNConv, TransformerConv, global_mean_pool
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTForImageClassification


MODEL_CANDIDATES = [
    "edadaltocg/vit-base-patch16-224-in21k-ft-cifar100",
    "aaraki/vit-base-patch16-224-in21k-finetuned-cifar100",
    "nateraw/vit-base-patch16-224-cifar100",
    "edumunozsala/vit_base-224-in21k-ft-cifar100",
]


@dataclass
class GraphSample:
    x_min: Tensor
    x_repr: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    layer_id: Optional[Tensor]
    layer_count: int
    y_err: Tensor
    vit_correct: Tensor
    vit_confidence: Tensor
    vit_margin: Tensor
    true_label: Tensor
    pred_label: Tensor


@dataclass
class Metrics:
    model: str
    node_features: str
    val_loss: float
    accuracy: float
    auroc: Optional[float]
    auprc: Optional[float]
    positives: int
    total: int


class GraphDataset(Dataset):
    def __init__(self, samples: Sequence[GraphSample], use_repr: bool):
        self.samples = list(samples)
        self.use_repr = use_repr

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        sample = self.samples[idx]
        x = sample.x_repr if self.use_repr else sample.x_min
        return Data(
            x=x.float(),
            edge_index=sample.edge_index.long(),
            edge_attr=sample.edge_attr.float(),
            layer_id=None if sample.layer_id is None else sample.layer_id.long(),
            layer_count=torch.tensor([sample.layer_count], dtype=torch.long),
            y=sample.y_err.float().view(1),
            confidence=sample.vit_confidence.float().view(1),
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_graphs(batch: Sequence[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]) -> Dict[str, Tensor]:
    xs, edge_indices, edge_attrs, ys, confidences = zip(*batch)
    node_offsets = []
    offset = 0
    for x in xs:
        node_offsets.append(offset)
        offset += x.shape[0]

    shifted_edges = [edge_index + node_offsets[i] for i, edge_index in enumerate(edge_indices)]
    batch_index = [torch.full((xs[i].shape[0],), i, dtype=torch.long) for i in range(len(xs))]

    return {
        "x": torch.cat(xs, dim=0).float(),
        "edge_index": torch.cat(shifted_edges, dim=1).long(),
        "edge_attr": torch.cat(edge_attrs, dim=0).float(),
        "batch": torch.cat(batch_index, dim=0).long(),
        "y": torch.stack(ys).float().view(-1),
        "confidence": torch.stack(confidences).float().view(-1),
    }


def pick_model(explicit_model: Optional[str]) -> str:
    if explicit_model:
        return explicit_model

    api = HfApi()
    for model_id in MODEL_CANDIDATES:
        try:
            api.model_info(model_id)
            return model_id
        except Exception:
            continue
    raise RuntimeError(
        "Could not find any known CIFAR-100 ViT model on HuggingFace. "
        "Pass --model-id with a ViTForImageClassification checkpoint fine-tuned for CIFAR-100."
    )


def load_vit(model_id: str, device: torch.device) -> Tuple[AutoImageProcessor, ViTForImageClassification]:
    try:
        processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    except OSError:
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)
    model = ViTForImageClassification.from_pretrained(model_id, attn_implementation="eager")
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model


def make_subset_indices(total: int, count: int, seed: int) -> List[int]:
    if count <= 0 or count >= total:
        return list(range(total))
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=min(count, total), replace=False)
    return indices.tolist()


class InMemoryCIFAR100Subset(Dataset):
    def __init__(self, rows: Sequence[Tuple[Image.Image, int]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        return self.rows[idx]


def load_hf_streamed_cifar100(count: int, seed: int) -> InMemoryCIFAR100Subset:
    from datasets import load_dataset

    stream = load_dataset(
        "uoft-cs/cifar100",
        split="test",
        streaming=True,
        cache_dir=str(PROJECT_ROOT / ".cache" / "huggingface" / "datasets"),
    )
    stream = stream.shuffle(seed=seed, buffer_size=max(count * 4, 100))
    rows: List[Tuple[Image.Image, int]] = []
    for row in stream.take(count):
        image = row.get("img", row.get("image"))
        label = row.get("fine_label", row.get("label"))
        if image is None or label is None:
            raise RuntimeError(f"Unexpected CIFAR-100 row keys: {sorted(row.keys())}")
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        rows.append((image.convert("RGB"), int(label)))
    return InMemoryCIFAR100Subset(rows)


def load_local_hf_parquet_cifar100(split: str, count: int, seed: int) -> InMemoryCIFAR100Subset:
    import pandas as pd

    parquet_path = PROJECT_ROOT / "data" / "hf_cifar100" / "cifar100" / f"{split}-00000-of-00001.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Missing {parquet_path}. Download it with the HuggingFace curl commands in the report, "
            "or use --data-source hf-stream."
        )
    frame = pd.read_parquet(parquet_path)
    indices = make_subset_indices(len(frame), count, seed)
    rows: List[Tuple[Image.Image, int]] = []
    for idx in indices:
        image_payload = frame.iloc[idx]["img"]
        if isinstance(image_payload, dict) and image_payload.get("bytes") is not None:
            image = Image.open(BytesIO(image_payload["bytes"]))
        else:
            image = Image.fromarray(np.asarray(image_payload))
        rows.append((image.convert("RGB"), int(frame.iloc[idx]["fine_label"])))
    return InMemoryCIFAR100Subset(rows)


def preprocess_batch(processor: AutoImageProcessor, images: Sequence[Image.Image], device: torch.device) -> Tensor:
    encoded = processor(images=list(images), return_tensors="pt")
    return encoded["pixel_values"].to(device)


def collate_images(batch: Sequence[Tuple[Image.Image, int]]) -> Tuple[List[Image.Image], Tensor]:
    images, labels = zip(*batch)
    return list(images), torch.as_tensor(labels, dtype=torch.long)


def scan_vit_correctness(
    dataset: Dataset,
    indices: Sequence[int],
    processor: AutoImageProcessor,
    model: ViTForImageClassification,
    device: torch.device,
    batch_size: int,
) -> Tuple[Dict[str, float], List[int], List[int]]:
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_images)
    correct_indices: List[int] = []
    wrong_indices: List[int] = []
    total_confidence = 0.0

    for batch_start, batch in enumerate(tqdm(loader, desc="scan ViT correctness")):
        images, labels = batch
        pixel_values = preprocess_batch(processor, images, device)
        labels_tensor = labels.long()
        with torch.no_grad():
            logits = model(pixel_values).logits.detach().cpu()
            probabilities = torch.softmax(logits, dim=-1)
            confidence, pred_label = probabilities.max(dim=-1)
        total_confidence += float(confidence.sum().item())
        for item_idx, prediction in enumerate(pred_label):
            original_index = indices[batch_start * batch_size + item_idx]
            if prediction.eq(labels_tensor[item_idx]):
                correct_indices.append(original_index)
            else:
                wrong_indices.append(original_index)

    total = len(correct_indices) + len(wrong_indices)
    summary = {
        "scan_total": total,
        "scan_correct": len(correct_indices),
        "scan_wrong": len(wrong_indices),
        "scan_accuracy": len(correct_indices) / max(total, 1),
        "scan_mean_confidence": total_confidence / max(total, 1),
    }
    return summary, correct_indices, wrong_indices


def balance_indices(
    correct_indices: Sequence[int],
    wrong_indices: Sequence[int],
    per_class: int,
    seed: int,
) -> List[int]:
    if not correct_indices or not wrong_indices:
        raise RuntimeError("Cannot balance dataset because the scan found only one ViT correctness class.")
    rng = random.Random(seed)
    correct = list(correct_indices)
    wrong = list(wrong_indices)
    rng.shuffle(correct)
    rng.shuffle(wrong)
    target = min(len(correct), len(wrong))
    if per_class > 0:
        target = min(target, per_class)
    selected = correct[:target] + wrong[:target]
    rng.shuffle(selected)
    return selected


def balanced_train_test_indices(
    correct_indices: Sequence[int],
    wrong_indices: Sequence[int],
    train_per_class: int,
    test_per_class: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    needed = train_per_class + test_per_class
    if len(correct_indices) < needed or len(wrong_indices) < needed:
        raise RuntimeError(
            f"Need at least {needed} correct and {needed} wrong examples, "
            f"but found {len(correct_indices)} correct and {len(wrong_indices)} wrong."
        )
    rng = random.Random(seed)
    correct = list(correct_indices)
    wrong = list(wrong_indices)
    rng.shuffle(correct)
    rng.shuffle(wrong)
    train_indices = correct[:train_per_class] + wrong[:train_per_class]
    test_indices = correct[train_per_class:needed] + wrong[train_per_class:needed]
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def write_scan_cache(path: Path, summary: Dict[str, float], correct_indices: Sequence[int], wrong_indices: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    payload["correct_indices"] = list(correct_indices)
    payload["wrong_indices"] = list(wrong_indices)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_scan_cache(path: Path) -> Tuple[Dict[str, float], List[int], List[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "correct_indices" not in payload or "wrong_indices" not in payload:
        raise RuntimeError(f"Scan cache {path} has counts only; rerun scan once to cache indices.")
    correct_indices = [int(idx) for idx in payload["correct_indices"]]
    wrong_indices = [int(idx) for idx in payload["wrong_indices"]]
    summary = {key: payload[key] for key in payload if key not in {"correct_indices", "wrong_indices"}}
    return summary, correct_indices, wrong_indices


def normalized_patch_features(num_tokens: int, attention: Tensor, layer_idx: int = 0, layer_count: int = 1) -> Tensor:
    # Node 0 is CLS. Remaining tokens are assumed to be a square patch grid.
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


def build_graph(
    attention: Tensor,
    hidden: Tensor,
    tau: float,
    topk: int,
    layer_idx: int = 0,
    layer_count: int = 1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # attention shape: heads, query/read token i, key/source token j.
    heads, num_tokens, _ = attention.shape
    max_attention = attention.max(dim=0).values
    keep = max_attention > tau
    keep.fill_diagonal_(False)

    if topk > 0:
        topk_keep = torch.zeros_like(keep)
        k = min(topk, num_tokens - 1)
        values, indices = max_attention.topk(k=k, dim=1)
        query_indices = torch.arange(num_tokens).unsqueeze(1).expand_as(indices)
        topk_keep[query_indices, indices] = values > 0
        topk_keep.fill_diagonal_(False)
        keep = keep | topk_keep

    query_i, key_j = keep.nonzero(as_tuple=True)
    if query_i.numel() == 0:
        # Fallback to a tiny graph if tau is too aggressive.
        query_i, key_j = (max_attention + torch.eye(num_tokens)).topk(k=1, dim=1).indices.view(-1), torch.arange(num_tokens)

    # Information flow is key/source j -> query/receiver i.
    edge_index = torch.stack([key_j, query_i], dim=0).long()
    edge_attr = attention[:, query_i, key_j].transpose(0, 1).contiguous().float()
    x_min = normalized_patch_features(num_tokens, attention, layer_idx=layer_idx, layer_count=layer_count).float()
    x_repr = torch.cat([x_min, hidden.float()], dim=1)
    return x_min, x_repr, edge_index, edge_attr


def build_all_layers_graph(attentions: Tensor, hidden_states: Sequence[Tensor], tau: float, topk: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    layer_count = attentions.shape[0]
    x_mins: List[Tensor] = []
    x_reprs: List[Tensor] = []
    edge_indices: List[Tensor] = []
    edge_attrs: List[Tensor] = []
    node_offset = 0
    edge_dim = attentions.shape[1]
    for layer_idx in range(layer_count):
        hidden = hidden_states[layer_idx + 1]
        x_min, x_repr, edge_index, edge_attr = build_graph(
            attentions[layer_idx],
            hidden,
            tau=tau,
            topk=topk,
            layer_idx=layer_idx,
            layer_count=layer_count,
        )
        x_mins.append(x_min)
        x_reprs.append(x_repr)
        edge_indices.append(edge_index + node_offset)
        edge_attrs.append(edge_attr)
        node_offset += x_min.shape[0]
    num_tokens = attentions.shape[-1]
    if layer_count > 1:
        temporal_src: List[Tensor] = []
        temporal_dst: List[Tensor] = []
        for layer_idx in range(layer_count - 1):
            source_offset = layer_idx * num_tokens
            target_offset = (layer_idx + 1) * num_tokens
            token_ids = torch.arange(num_tokens, dtype=torch.long)
            temporal_src.append(source_offset + token_ids)
            temporal_dst.append(target_offset + token_ids)
        temporal_edge_index = torch.stack([torch.cat(temporal_src), torch.cat(temporal_dst)], dim=0)
        temporal_edge_attr = torch.zeros((temporal_edge_index.shape[1], edge_dim), dtype=torch.float32)
        temporal_edge_attr[:, 0] = 1.0
        edge_indices.append(temporal_edge_index)
        edge_attrs.append(temporal_edge_attr)
    return torch.cat(x_mins, dim=0), torch.cat(x_reprs, dim=0), torch.cat(edge_indices, dim=1), torch.cat(edge_attrs, dim=0)


def build_layer_sequence_graph(attentions: Tensor, hidden_states: Sequence[Tensor], tau: float, topk: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
    layer_count = attentions.shape[0]
    x_mins: List[Tensor] = []
    x_reprs: List[Tensor] = []
    edge_indices: List[Tensor] = []
    edge_attrs: List[Tensor] = []
    layer_ids: List[Tensor] = []
    node_offset = 0
    for layer_idx in range(layer_count):
        hidden = hidden_states[layer_idx + 1]
        x_min, x_repr, edge_index, edge_attr = build_graph(
            attentions[layer_idx],
            hidden,
            tau=tau,
            topk=topk,
            layer_idx=layer_idx,
            layer_count=layer_count,
        )
        x_mins.append(x_min)
        x_reprs.append(x_repr)
        edge_indices.append(edge_index + node_offset)
        edge_attrs.append(edge_attr)
        layer_ids.append(torch.full((x_min.shape[0],), layer_idx, dtype=torch.long))
        node_offset += x_min.shape[0]
    return (
        torch.cat(x_mins, dim=0),
        torch.cat(x_reprs, dim=0),
        torch.cat(edge_indices, dim=1),
        torch.cat(edge_attrs, dim=0),
        torch.cat(layer_ids, dim=0),
        layer_count,
    )


def extract_graphs(
    dataset: Dataset,
    indices: Sequence[int],
    processor: AutoImageProcessor,
    model: ViTForImageClassification,
    device: torch.device,
    batch_size: int,
    tau: float,
    topk: int,
    attention_layers: str,
    max_batches: Optional[int] = None,
) -> Tuple[List[GraphSample], Dict[str, float]]:
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_images)
    samples: List[GraphSample] = []
    correct_count = 0
    total_count = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="extract ViT graphs")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, labels = batch
        pixel_values = preprocess_batch(processor, images, device)
        labels_tensor = torch.as_tensor(labels, dtype=torch.long)

        with torch.no_grad():
            outputs = model(pixel_values, output_attentions=True, output_hidden_states=True)
            logits = outputs.logits.detach().cpu()
            probabilities = torch.softmax(logits, dim=-1)
            confidence, pred_label = probabilities.max(dim=-1)
            top2 = probabilities.topk(k=2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            if attention_layers in {"all", "sequence"}:
                attentions = torch.stack([attention.detach().cpu() for attention in outputs.attentions], dim=1)
                hidden_states = [hidden.detach().cpu() for hidden in outputs.hidden_states]
            else:
                attentions = outputs.attentions[-1].detach().cpu()
                hidden_states = outputs.hidden_states[-1].detach().cpu()

        for item_idx in range(logits.shape[0]):
            true_label = labels_tensor[item_idx]
            prediction = pred_label[item_idx]
            correct = prediction.eq(true_label)
            layer_id: Optional[Tensor] = None
            layer_count = 1
            if attention_layers == "sequence":
                item_hidden_states = [hidden[item_idx] for hidden in hidden_states]
                x_min, x_repr, edge_index, edge_attr, layer_id, layer_count = build_layer_sequence_graph(
                    attentions[item_idx], item_hidden_states, tau=tau, topk=topk
                )
            elif attention_layers == "all":
                item_hidden_states = [hidden[item_idx] for hidden in hidden_states]
                x_min, x_repr, edge_index, edge_attr = build_all_layers_graph(
                    attentions[item_idx], item_hidden_states, tau=tau, topk=topk
                )
                layer_count = attentions[item_idx].shape[0]
            else:
                x_min, x_repr, edge_index, edge_attr = build_graph(
                    attentions[item_idx], hidden_states[item_idx], tau=tau, topk=topk
                )
                layer_id = torch.zeros((x_min.shape[0],), dtype=torch.long)
            samples.append(
                GraphSample(
                    x_min=x_min,
                    x_repr=x_repr,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    layer_id=layer_id,
                    layer_count=layer_count,
                    y_err=(~correct).float(),
                    vit_correct=correct.float(),
                    vit_confidence=confidence[item_idx],
                    vit_margin=margin[item_idx],
                    true_label=true_label,
                    pred_label=prediction,
                )
            )
            correct_count += int(correct.item())
            total_count += 1

    summary = {
        "vit_accuracy": correct_count / max(total_count, 1),
        "vit_errors": total_count - correct_count,
        "total": total_count,
        "mean_edges": float(np.mean([sample.edge_index.shape[1] for sample in samples])) if samples else 0.0,
    }
    return samples, summary


class GraphClassifier(nn.Module):
    def __init__(self, kind: str, in_dim: int, edge_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.kind = kind
        if kind not in {"gcn", "nnconv_edge", "transformer_edge", "gatv2_edge"}:
            raise ValueError(f"Unknown model kind: {kind}")
        dims = [in_dim] + [hidden_dim] * layers
        pyg_layers: List[nn.Module] = []
        for idx in range(layers):
            layer_in = dims[idx]
            layer_out = dims[idx + 1]
            if kind == "gcn":
                pyg_layers.append(GCNConv(layer_in, layer_out))
            elif kind == "nnconv_edge":
                edge_network = nn.Sequential(
                    nn.Linear(edge_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, layer_in * layer_out),
                )
                pyg_layers.append(NNConv(layer_in, layer_out, edge_network, aggr="mean"))
            elif kind == "transformer_edge":
                pyg_layers.append(TransformerConv(layer_in, layer_out, heads=2, concat=False, edge_dim=edge_dim))
            else:
                pyg_layers.append(GATv2Conv(layer_in, layer_out, heads=2, concat=False, edge_dim=edge_dim))
        self.layers = nn.ModuleList(pyg_layers)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor, layer_id: Optional[Tensor] = None) -> Tensor:
        for layer, norm in zip(self.layers, self.norms):
            if self.kind == "gcn":
                x = layer(x, edge_index)
            else:
                x = layer(x, edge_index, edge_attr)
            x = torch.relu(norm(x))
        graph_embedding = global_mean_pool(x, batch)
        return self.head(graph_embedding).view(-1)


class LayerSequenceClassifier(nn.Module):
    def __init__(self, kind: str, in_dim: int, edge_dim: int, hidden_dim: int, layers: int, dropout: float, layer_count: int):
        super().__init__()
        if kind != "layer_sequence_transformer":
            raise ValueError(f"Unknown layer-sequence model kind: {kind}")
        self.layer_count = layer_count
        dims = [in_dim] + [hidden_dim] * layers
        self.layers = nn.ModuleList(
            [
                TransformerConv(dims[idx], dims[idx + 1], heads=2, concat=False, edge_dim=edge_dim)
                for idx in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * layer_count, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor, layer_id: Optional[Tensor] = None) -> Tensor:
        if layer_id is None:
            raise RuntimeError("LayerSequenceClassifier requires per-node layer_id.")
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, edge_index, edge_attr)
            x = torch.relu(norm(x))
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
        layer_graph_id = batch * self.layer_count + layer_id
        layer_embeddings = global_mean_pool(x, layer_graph_id, size=graph_count * self.layer_count)
        ordered = layer_embeddings.view(graph_count, self.layer_count, -1).reshape(graph_count, -1)
        return self.decoder(ordered).view(-1)


def split_samples(samples: Sequence[GraphSample], val_fraction: float, seed: int) -> Tuple[List[GraphSample], List[GraphSample]]:
    rng = random.Random(seed)
    positive_indices = [idx for idx, sample in enumerate(samples) if sample.y_err.item() > 0.5]
    negative_indices = [idx for idx, sample in enumerate(samples) if sample.y_err.item() <= 0.5]
    rng.shuffle(positive_indices)
    rng.shuffle(negative_indices)
    if positive_indices and negative_indices:
        positive_val_count = max(1, int(len(positive_indices) * val_fraction))
        negative_val_count = max(1, int(len(negative_indices) * val_fraction))
        val_indices = set(positive_indices[:positive_val_count] + negative_indices[:negative_val_count])
    else:
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        val_count = max(1, int(len(indices) * val_fraction))
        val_indices = set(indices[:val_count])
    train = [sample for idx, sample in enumerate(samples) if idx not in val_indices]
    val = [sample for idx, sample in enumerate(samples) if idx in val_indices]
    return train, val


def batch_to_device(batch: Data, device: torch.device) -> Data:
    return batch.to(device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    probabilities: List[float] = []
    labels: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, getattr(batch, "layer_id", None))
            loss = criterion(logits, batch.y.view(-1))
            probs = torch.sigmoid(logits)
            total_loss += float(loss.item()) * batch.y.numel()
            total += batch.y.numel()
            probabilities.extend(probs.detach().cpu().numpy().tolist())
            labels.extend(batch.y.view(-1).detach().cpu().numpy().tolist())
    return total_loss / max(total, 1), np.asarray(probabilities), np.asarray(labels)


def train_one(
    model_kind: str,
    node_features: str,
    train_samples: Sequence[GraphSample],
    val_samples: Sequence[GraphSample],
    args: argparse.Namespace,
    device: torch.device,
) -> Metrics:
    use_repr = node_features == "minimal+repr"
    train_dataset = GraphDataset(train_samples, use_repr=use_repr)
    val_dataset = GraphDataset(val_samples, use_repr=use_repr)
    train_loader = PyGDataLoader(train_dataset, batch_size=args.gnn_batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=args.gnn_batch_size, shuffle=False)

    first_sample = train_dataset[0]
    in_dim = first_sample.x.shape[1]
    edge_dim = first_sample.edge_attr.shape[1]
    if model_kind == "layer_sequence_transformer":
        layer_count = int(first_sample.layer_count.view(-1)[0].item())
        model = LayerSequenceClassifier(
            kind=model_kind,
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=args.hidden_dim,
            layers=args.gnn_layers,
            dropout=args.dropout,
            layer_count=layer_count,
        ).to(device)
    else:
        model = GraphClassifier(
            kind=model_kind,
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=args.hidden_dim,
            layers=args.gnn_layers,
            dropout=args.dropout,
        ).to(device)

    positive_count = sum(int(sample.y_err.item()) for sample in train_samples)
    negative_count = max(len(train_samples) - positive_count, 1)
    pos_weight = torch.tensor([negative_count / max(positive_count, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    train_bar = tqdm(total=args.epochs * len(train_loader), desc=f"train {model_kind}/{node_features}")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, getattr(batch, "layer_id", None))
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * batch.y.numel()
            seen += batch.y.numel()
            train_bar.update(1)
            train_bar.set_postfix(epoch=epoch + 1, train_loss=f"{running_loss / max(seen, 1):.4f}")
        val_loss, _, _ = evaluate(model, val_loader, device, criterion)
        train_bar.set_postfix(epoch=epoch + 1, train_loss=f"{running_loss / max(seen, 1):.4f}", val_loss=f"{val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    train_bar.close()

    if best_state is not None:
        model.load_state_dict(best_state)
    val_loss, probabilities, labels = evaluate(model, val_loader, device, criterion)
    predictions = probabilities >= 0.5
    accuracy = accuracy_score(labels, predictions)
    positives = int(labels.sum())
    auroc = roc_auc_score(labels, probabilities) if len(np.unique(labels)) > 1 else None
    auprc = average_precision_score(labels, probabilities) if len(np.unique(labels)) > 1 else None
    return Metrics(
        model=model_kind,
        node_features=node_features,
        val_loss=float(val_loss),
        accuracy=float(accuracy),
        auroc=None if auroc is None else float(auroc),
        auprc=None if auprc is None else float(auprc),
        positives=positives,
        total=int(labels.shape[0]),
    )


def confidence_baselines(samples: Sequence[GraphSample]) -> Dict[str, Optional[float]]:
    labels = np.asarray([sample.y_err.item() for sample in samples])
    confidence = np.asarray([sample.vit_confidence.item() for sample in samples])
    uncertainty = 1.0 - confidence
    if len(np.unique(labels)) <= 1:
        return {"msp_auroc": None, "msp_auprc": None}
    return {
        "msp_auroc": float(roc_auc_score(labels, uncertainty)),
        "msp_auprc": float(average_precision_score(labels, uncertainty)),
    }


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    model_id: str,
    device: torch.device,
    extraction_summary: Dict[str, float],
    metrics: Sequence[Metrics],
    baselines: Dict[str, Optional[float]],
    elapsed: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_id,
        "device": str(device),
        "args": vars(args),
        "extraction_summary": extraction_summary,
        "confidence_baselines": baselines,
        "gnn_metrics": [asdict(metric) for metric in metrics],
        "elapsed_seconds": elapsed,
    }
    (output_dir / "poc_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Frozen ViT Attention Graph POC Report",
        "",
        "## Scope",
        "This POC follows the project proposal discussed in the conversation: freeze a ViT, convert attention into token graphs, and train only graph-based failure detectors above it.",
        "",
        "## Run Configuration",
        f"- Project root: `{PROJECT_ROOT}`",
        f"- HuggingFace model: `{model_id}`",
        f"- Device: `{device}`",
        f"- CIFAR-100 samples requested: `{args.samples}`",
        f"- Data source: `{args.data_source}` / split `{args.split}`",
        f"- Balanced by ViT correctness: `{args.balance_by_vit}`",
        f"- GNN architecture: `{args.model_kind}`",
        f"- Attention layers: `{args.attention_layers}`",
        f"- Edge rule: keep max-head attention > `{args.tau}` plus top-`{args.topk}` incoming edges per query token",
        f"- Epochs per GNN: `{args.epochs}`",
        "",
        "## Frozen ViT Check",
        f"- Evaluated samples: `{int(extraction_summary['total'])}`",
        f"- Frozen ViT accuracy on sampled CIFAR-100: `{extraction_summary['vit_accuracy']:.4f}`",
        f"- Frozen ViT errors / positives: `{int(extraction_summary['vit_errors'])}`",
        f"- Frozen ViT correct examples: `{int(extraction_summary['vit_correct'])}`",
        f"- Mean graph edges: `{extraction_summary['mean_edges']:.1f}`",
        "",
        "## Baseline",
        f"- Maximum softmax probability AUROC: `{baselines['msp_auroc']}`",
        f"- Maximum softmax probability AUPRC: `{baselines['msp_auprc']}`",
        "",
        "## GNN Results",
        "| Model | Node Features | AUROC | AUPRC | Accuracy | Val Loss | Positives/Total |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        lines.append(
            f"| {metric.model} | {metric.node_features} | {metric.auroc} | {metric.auprc} | "
            f"{metric.accuracy:.4f} | {metric.val_loss:.4f} | {metric.positives}/{metric.total} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- This is intentionally a small Mac-friendly POC, not the full proposal experiment.",
            "- It uses only the last attention layer to keep runtime and memory practical on a MacBook Air.",
            "- The full project should extend this to all layers, corruption splits, risk-coverage metrics, and stronger baselines.",
            f"- Elapsed wall time: `{elapsed:.1f}` seconds.",
        ]
    )
    (output_dir / "poc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=None, help="HF ViTForImageClassification checkpoint. Auto-picks known CIFAR-100 ViT if omitted.")
    parser.add_argument("--samples", type=int, default=0, help="CIFAR-100 examples to scan; 0 means the full selected split for hf-local.")
    parser.add_argument(
        "--data-source",
        choices=["hf-local", "hf-stream", "torchvision"],
        default="hf-local",
        help="hf-local reads data/hf_cifar100 parquet files; hf-stream fetches requested examples; torchvision downloads the full Toronto archive.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "train"],
        default="test",
        help="CIFAR-100 split to use for hf-local. Torchvision and hf-stream use test.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scan-batch-size", type=int, default=8)
    parser.add_argument("--vit-batch-size", type=int, default=2)
    parser.add_argument("--gnn-batch-size", type=int, default=4)
    parser.add_argument("--attention-layers", choices=["last", "all", "sequence"], default="last", help="Use the last ViT layer, one connected all-layer graph, or proposal-style sequence of layer graphs.")
    parser.add_argument("--node-features", choices=["both", "minimal", "minimal+repr"], default="both", help="Which node feature condition(s) to train.")
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.30)
    parser.add_argument("--scan-only", action="store_true", help="Only count ViT correct/wrong examples; do not extract graphs or train GNNs.")
    parser.add_argument("--scan-cache", default="reports_vit_scan_test/vit_scan_with_indices.json", help="Path for cached ViT correct/wrong indices, relative to the project root unless absolute.")
    parser.add_argument("--balance-by-vit", action="store_true", help="Use equal numbers of ViT-correct and ViT-wrong examples for graph training.")
    parser.add_argument("--balanced-per-class", type=int, default=100, help="Max correct and wrong examples to use when --balance-by-vit is set; 0 uses all available pairs.")
    parser.add_argument("--balanced-train-per-class", type=int, default=0, help="Exact correct and wrong examples per class for training; enables fixed balanced train/test when > 0.")
    parser.add_argument("--balanced-test-per-class", type=int, default=0, help="Exact correct and wrong examples per class for held-out test when fixed balanced train/test is enabled.")
    parser.add_argument(
        "--model-kind",
        choices=["gcn", "nnconv_edge", "transformer_edge", "gatv2_edge", "layer_sequence_transformer"],
        default="transformer_edge",
        help="Single PyG architecture to train for the POC.",
    )
    parser.add_argument("--all-models", action="store_true", help="Run the full architecture sweep instead of only --model-kind.")
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


def main() -> None:
    start = time.time()
    args = parse_args()
    seed_everything(args.seed)
    data_root = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / args.output_dir
    device = choose_device()

    model_id = pick_model(args.model_id)
    print(f"Using model: {model_id}")
    print(f"Using device: {device}")
    processor, model = load_vit(model_id, device)

    if args.data_source == "hf-local":
        dataset = load_local_hf_parquet_cifar100(args.split, args.samples, args.seed)
        indices = list(range(len(dataset)))
    elif args.data_source == "hf-stream":
        dataset = load_hf_streamed_cifar100(args.samples, args.seed)
        indices = list(range(len(dataset)))
    else:
        # torchvision is only needed for this data source; keep it off the module import path.
        from torchvision.datasets import CIFAR100

        dataset = CIFAR100(root=str(data_root), train=False, download=True)
        indices = make_subset_indices(len(dataset), args.samples, args.seed)

    scan_cache = Path(args.scan_cache)
    if not scan_cache.is_absolute():
        scan_cache = PROJECT_ROOT / scan_cache
    if scan_cache.exists():
        scan_summary, correct_indices, wrong_indices = read_scan_cache(scan_cache)
        print(f"Loaded ViT scan cache from {scan_cache}")
    else:
        scan_summary, correct_indices, wrong_indices = scan_vit_correctness(
            dataset=dataset,
            indices=indices,
            processor=processor,
            model=model,
            device=device,
            batch_size=args.scan_batch_size,
        )
        write_scan_cache(scan_cache, scan_summary, correct_indices, wrong_indices)
        print(f"Wrote ViT scan cache to {scan_cache}")
    print(
        "ViT scan: "
        f"{scan_summary['scan_correct']} correct, {scan_summary['scan_wrong']} wrong, "
        f"accuracy={scan_summary['scan_accuracy']:.4f} over {scan_summary['scan_total']} examples"
    )
    if args.scan_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_scan_cache(output_dir / "vit_scan.json", scan_summary, correct_indices, wrong_indices)
        print(f"Wrote scan summary to {output_dir / 'vit_scan.json'}")
        return

    fixed_split = args.balanced_train_per_class > 0 or args.balanced_test_per_class > 0
    train_indices: List[int] = []
    test_indices: List[int] = []
    if fixed_split:
        if args.balanced_train_per_class <= 0 or args.balanced_test_per_class <= 0:
            raise RuntimeError("Use both --balanced-train-per-class and --balanced-test-per-class for fixed balanced splits.")
        train_indices, test_indices = balanced_train_test_indices(
            correct_indices,
            wrong_indices,
            args.balanced_train_per_class,
            args.balanced_test_per_class,
            args.seed,
        )
        indices = train_indices + test_indices
        print(
            "Fixed balanced split: "
            f"train={args.balanced_train_per_class} correct + {args.balanced_train_per_class} wrong, "
            f"test={args.balanced_test_per_class} correct + {args.balanced_test_per_class} wrong"
        )
    if args.balance_by_vit:
        indices = balance_indices(correct_indices, wrong_indices, args.balanced_per_class, args.seed)
        print(f"Balanced extraction set: {len(indices) // 2} correct + {len(indices) // 2} wrong = {len(indices)} examples")

    samples, extraction_summary = extract_graphs(
        dataset=dataset,
        indices=indices,
        processor=processor,
        model=model,
        device=device,
        batch_size=args.vit_batch_size,
        tau=args.tau,
        topk=args.topk,
        attention_layers=args.attention_layers,
    )
    extraction_summary.update(scan_summary)
    extraction_summary["vit_correct"] = extraction_summary["total"] - extraction_summary["vit_errors"]
    if len(samples) < 20:
        raise RuntimeError("Too few samples were extracted for a useful POC.")

    if fixed_split:
        train_samples = samples[: len(train_indices)]
        val_samples = samples[len(train_indices) :]
    else:
        train_samples, val_samples = split_samples(samples, args.val_fraction, args.seed)
    baselines = confidence_baselines(val_samples)
    metrics: List[Metrics] = []
    model_kinds = ["gcn", "nnconv_edge", "transformer_edge", "gatv2_edge", "layer_sequence_transformer"] if args.all_models else [args.model_kind]
    node_feature_options = ["minimal", "minimal+repr"] if args.node_features == "both" else [args.node_features]
    for node_features in node_feature_options:
        for model_kind in model_kinds:
            print(f"Training {model_kind} with {node_features}")
            metric = train_one(model_kind, node_features, train_samples, val_samples, args, device)
            metrics.append(metric)
            print(metric)

    elapsed = time.time() - start
    write_report(output_dir, args, model_id, device, extraction_summary, metrics, baselines, elapsed)
    print(f"Wrote report to {output_dir / 'poc_report.md'}")
    print(f"Wrote JSON to {output_dir / 'poc_results.json'}")


if __name__ == "__main__":
    main()
