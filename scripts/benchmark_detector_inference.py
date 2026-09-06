#!/usr/bin/env python3
"""Benchmark prepared-feature detector latency separately from dataset feature loading."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polygraph.data.splits import SplitPlan
from polygraph.data.storage import AttentionGraphDataset, GraphStore, LastFourGraphDataset
from polygraph.training.train import load_checkpoint


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=Path("data/graph_dataset/store"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--records", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batches", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_checkpoint(args.checkpoint, device)
    plan, store = SplitPlan.load(args.plan), GraphStore(args.store, cache_shards=1)
    keys = plan.splits["test"][:args.records]
    if getattr(config, "multilayer_mode", "none") != "none":
        dataset = LastFourGraphDataset(store, keys, config.multilayer_mode)
    else:
        node_features = getattr(config, "node_features", "base")
        if getattr(config, "hidden", False) and node_features == "base": node_features = "hidden"
        dataset = AttentionGraphDataset(
            store, config.layers, keys, tau=config.tau, top_k=config.top_k,
            hidden_dir=args.store.parent / "hidden12" if node_features == "hidden" else None,
            edge_features=getattr(config, "edge_features", "attention"),
            message_stats_dir=(args.store.parent / "sidecars/message_stats_l11"
                               if getattr(config, "edge_features", "attention") != "attention" else None),
            omit_edges=getattr(config, "architecture", "") == "hidden_token_set")
    batch_size = min(config.batch_size, max(1, len(dataset)))
    start = time.perf_counter()
    host_batches = list(DataLoader(dataset, batch_size=batch_size, shuffle=False))
    feature_seconds = time.perf_counter() - start
    host_batches = host_batches[:args.batches]
    gpu_batches = [batch.to(device) for batch in host_batches]
    with torch.no_grad():
        for _ in range(args.warmup):
            for batch in gpu_batches: model(batch)
        synchronize(); times = []
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        for batch in gpu_batches:
            start = time.perf_counter(); model(batch); synchronize()
            times.append(time.perf_counter() - start)
    records = sum(batch.num_graphs for batch in gpu_batches)
    result = {
        "checkpoint": str(args.checkpoint), "prepared_records": records,
        "batch_size": batch_size, "batches": len(gpu_batches),
        "detector_seconds": float(sum(times)),
        "detector_ms_per_record": 1000 * float(sum(times)) / records,
        "feature_loading_seconds_for_requested_records": feature_seconds,
        "feature_loading_ms_per_record": 1000 * feature_seconds / len(dataset),
        "feature_loading_scope": "stored-feature loading and graph construction; excludes ViT forward",
        "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated() / 1024 ** 2
                               if device.type == "cuda" else None),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "latency_samples_seconds": times,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n"); temporary.replace(args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
