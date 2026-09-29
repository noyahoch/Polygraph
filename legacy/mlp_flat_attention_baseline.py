#!/usr/bin/env python3
"""Flat-attention MLP baseline for the frozen ViT failure-detection POC.

This script reuses the exact cached ViT-correct / ViT-wrong split from
poc_gnn_vit_cifar100.py, extracts dense last-layer attention tensors, trains a
small MLP directly over flattened attention matrices, and writes a comparison
against the saved GNN run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path.cwd().resolve()
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "datasets"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from poc_gnn_vit_cifar100 import (
    balanced_train_test_indices,
    choose_device,
    collate_images,
    load_local_hf_parquet_cifar100,
    load_vit,
    pick_model,
    preprocess_batch,
    read_scan_cache,
    seed_everything,
)


@dataclass
class MLPBaselineResult:
    model: str
    input_features: int
    hidden_dim: int
    parameters: int
    extraction_seconds: float
    training_seconds: float
    test_loss: float
    accuracy: float
    auroc: float
    auprc: float
    positives: int
    total: int


class FlatAttentionDataset(Dataset):
    def __init__(self, feature_path: Path, label_path: Path, indices: Sequence[int], feature_shape: Tuple[int, int]):
        self.features = np.memmap(feature_path, mode="r", dtype=np.float16, shape=feature_shape)
        self.labels = np.load(label_path)
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        row_idx = self.indices[idx]
        x = torch.from_numpy(np.asarray(self.features[row_idx], dtype=np.float32))
        y = torch.tensor(self.labels[row_idx], dtype=torch.float32)
        return x, y


class FlatAttentionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).view(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def extract_flat_attention_cache(
    dataset: Dataset,
    selected_indices: Sequence[int],
    processor,
    model,
    device: torch.device,
    batch_size: int,
    cache_dir: Path,
    force: bool,
) -> Tuple[Path, Path, Dict[str, int], float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    feature_path = cache_dir / "flat_attention.float16.memmap"
    label_path = cache_dir / "labels.npy"

    if not force and metadata_path.exists() and feature_path.exists() and label_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("num_examples") == len(selected_indices):
            return feature_path, label_path, metadata, 0.0

    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_images)
    start = time.time()
    features = None
    labels_out: List[float] = []
    write_offset = 0
    flat_dim = None

    for images, labels in tqdm(loader, desc="extract flat attentions"):
        pixel_values = preprocess_batch(processor, images, device)
        labels_tensor = labels.long()
        with torch.no_grad():
            outputs = model(pixel_values, output_attentions=True)
            logits = outputs.logits.detach().cpu()
            predictions = logits.argmax(dim=-1)
            attentions = outputs.attentions[-1].detach().cpu().numpy().astype(np.float16)

        batch_flat = attentions.reshape(attentions.shape[0], -1)
        if features is None:
            flat_dim = int(batch_flat.shape[1])
            features = np.memmap(feature_path, mode="w+", dtype=np.float16, shape=(len(selected_indices), flat_dim))
        batch_size_actual = batch_flat.shape[0]
        features[write_offset : write_offset + batch_size_actual] = batch_flat
        labels_out.extend((predictions.ne(labels_tensor)).float().numpy().tolist())
        write_offset += batch_size_actual

    assert features is not None and flat_dim is not None
    features.flush()
    np.save(label_path, np.asarray(labels_out, dtype=np.float32))
    metadata = {
        "num_examples": len(selected_indices),
        "input_dim": flat_dim,
        "dtype": "float16",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return feature_path, label_path, metadata, time.time() - start


def evaluate_mlp(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    probabilities: List[float] = []
    labels: List[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits)
            total_loss += float(loss.item()) * y.numel()
            total += y.numel()
            probabilities.extend(probs.detach().cpu().numpy().tolist())
            labels.extend(y.detach().cpu().numpy().tolist())
    return total_loss / max(total, 1), np.asarray(probabilities), np.asarray(labels)


def train_mlp(
    feature_path: Path,
    label_path: Path,
    metadata: Dict[str, int],
    train_count: int,
    args: argparse.Namespace,
    device: torch.device,
) -> MLPBaselineResult:
    train_indices = list(range(train_count))
    test_indices = list(range(train_count, metadata["num_examples"]))
    feature_shape = (metadata["num_examples"], metadata["input_dim"])
    train_dataset = FlatAttentionDataset(feature_path, label_path, train_indices, feature_shape)
    test_dataset = FlatAttentionDataset(feature_path, label_path, test_indices, feature_shape)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FlatAttentionMLP(metadata["input_dim"], args.hidden_dim, args.dropout).to(device)
    parameter_count = count_parameters(model)
    labels = np.load(label_path)
    train_labels = labels[:train_count]
    positive_count = int(train_labels.sum())
    negative_count = max(train_count - positive_count, 1)
    pos_weight = torch.tensor([negative_count / max(positive_count, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_loss = float("inf")
    start = time.time()
    train_bar = tqdm(total=args.epochs * len(train_loader), desc="train flat-attention MLP")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * y.numel()
            seen += y.numel()
            train_bar.update(1)
            train_bar.set_postfix(epoch=epoch + 1, train_loss=f"{running_loss / max(seen, 1):.4f}")
        test_loss, _, _ = evaluate_mlp(model, test_loader, device, criterion)
        train_bar.set_postfix(epoch=epoch + 1, train_loss=f"{running_loss / max(seen, 1):.4f}", test_loss=f"{test_loss:.4f}")
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    train_bar.close()
    training_seconds = time.time() - start

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, probabilities, labels_np = evaluate_mlp(model, test_loader, device, criterion)
    predictions = probabilities >= 0.5
    return MLPBaselineResult(
        model="flat_attention_mlp",
        input_features=metadata["input_dim"],
        hidden_dim=args.hidden_dim,
        parameters=parameter_count,
        extraction_seconds=0.0,
        training_seconds=training_seconds,
        test_loss=float(test_loss),
        accuracy=float(accuracy_score(labels_np, predictions)),
        auroc=float(roc_auc_score(labels_np, probabilities)),
        auprc=float(average_precision_score(labels_np, probabilities)),
        positives=int(labels_np.sum()),
        total=int(labels_np.shape[0]),
    )


def write_comparison(output_dir: Path, result: MLPBaselineResult, gnn_results_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gnn_payload = json.loads(gnn_results_path.read_text(encoding="utf-8")) if gnn_results_path.exists() else None
    payload = {
        "mlp_baseline": asdict(result),
        "gnn_reference_path": str(gnn_results_path),
        "gnn_reference": gnn_payload,
    }
    (output_dir / "flat_attention_mlp_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Flat Attention MLP Baseline",
        "",
        "## Baseline Result",
        f"- Model: `{result.model}`",
        f"- Input features: `{result.input_features}` flattened last-layer attention values",
        f"- Hidden dim: `{result.hidden_dim}`",
        f"- Trainable parameters: `{result.parameters}`",
        f"- Attention extraction/cache time: `{result.extraction_seconds:.1f}` seconds",
        f"- MLP training time: `{result.training_seconds:.1f}` seconds",
        f"- Test AUROC: `{result.auroc:.4f}`",
        f"- Test AUPRC: `{result.auprc:.4f}`",
        f"- Test accuracy: `{result.accuracy:.4f}`",
        f"- Test loss: `{result.test_loss:.4f}`",
        f"- Positives/total: `{result.positives}/{result.total}`",
        "",
        "## GNN Reference",
    ]
    if gnn_payload is None:
        lines.append(f"- Missing GNN reference JSON at `{gnn_results_path}`")
    else:
        lines.append(f"- GNN total elapsed time, including graph extraction and both feature conditions: `{gnn_payload['elapsed_seconds']:.1f}` seconds")
        for metric in gnn_payload["gnn_metrics"]:
            lines.append(
                f"- `{metric['model']}` / `{metric['node_features']}`: "
                f"AUROC `{metric['auroc']:.4f}`, AUPRC `{metric['auprc']:.4f}`, "
                f"accuracy `{metric['accuracy']:.4f}`, loss `{metric['val_loss']:.4f}`"
            )
    lines.extend(
        [
            "",
            "## Caveat",
            "A direct flattened-attention MLP cannot truly match the GNN parameter count: the input dimension alone is hundreds of thousands of values, so even a one-hidden-unit MLP has about one parameter per attention value.",
        ]
    )
    (output_dir / "flat_attention_mlp_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--scan-cache", default="reports_vit_scan_test/vit_scan_with_indices.json")
    parser.add_argument("--balanced-train-per-class", type=int, default=750)
    parser.add_argument("--balanced-test-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--vit-batch-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--cache-dir", default="data/flat_attention_baseline/test_750_100")
    parser.add_argument("--gnn-results", default="reports_balanced_750_100_transformer/poc_results.json")
    parser.add_argument("--output-dir", default="reports_flat_attention_mlp")
    parser.add_argument("--model-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device()
    model_id = pick_model(args.model_id)
    print(f"Using model: {model_id}")
    print(f"Using device: {device}")

    dataset = load_local_hf_parquet_cifar100(args.split, 0, args.seed)
    scan_cache = Path(args.scan_cache)
    if not scan_cache.is_absolute():
        scan_cache = PROJECT_ROOT / scan_cache
    _, correct_indices, wrong_indices = read_scan_cache(scan_cache)
    train_indices, test_indices = balanced_train_test_indices(
        correct_indices,
        wrong_indices,
        args.balanced_train_per_class,
        args.balanced_test_per_class,
        args.seed,
    )
    selected_indices = train_indices + test_indices
    print(
        "Fixed balanced split: "
        f"train={args.balanced_train_per_class} correct + {args.balanced_train_per_class} wrong, "
        f"test={args.balanced_test_per_class} correct + {args.balanced_test_per_class} wrong"
    )

    processor, vit = load_vit(model_id, device)
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    feature_path, label_path, metadata, extraction_seconds = extract_flat_attention_cache(
        dataset,
        selected_indices,
        processor,
        vit,
        device,
        args.vit_batch_size,
        cache_dir,
        args.force_extract,
    )
    result = train_mlp(
        feature_path,
        label_path,
        metadata,
        len(train_indices),
        args,
        device,
    )
    result.extraction_seconds = extraction_seconds
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    gnn_results = Path(args.gnn_results)
    if not gnn_results.is_absolute():
        gnn_results = PROJECT_ROOT / gnn_results
    write_comparison(output_dir, result, gnn_results)
    print(result)
    print(f"Wrote report to {output_dir / 'flat_attention_mlp_report.md'}")
    print(f"Wrote JSON to {output_dir / 'flat_attention_mlp_results.json'}")


if __name__ == "__main__":
    main()
