#!/usr/bin/env python3
"""Bounded CUDA/dataset profile for M5-level model families."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polygraph.data.splits import SplitPlan
from polygraph.data.storage import AttentionGraphDataset, GraphStore
from polygraph.training.train import TrainConfig, build_model


def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def timed(fn):
    sync(); start = time.perf_counter(); value = fn(); sync()
    return value, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=Path("data/graph_dataset/store"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--family", default="transformerconv")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    plan, store = SplitPlan.load(args.plan), GraphStore(args.store)
    dataset = AttentionGraphDataset(store, [11], plan.splits["train"][:args.batch_size],
                                    hidden_dir=args.store.parent / "hidden12",
                                    edge_features="evidence_flow",
                                    message_stats_dir=args.store.parent / "sidecars/message_stats_l11")
    # Cold item includes graph-shard/sidecar loading and feature construction. Warm item
    # isolates construction with all corresponding shards resident in bounded caches.
    item, cold = timed(lambda: dataset[0])
    _, warm = timed(lambda: dataset[0])
    items, item_window = timed(lambda: [dataset[i] for i in range(len(dataset))])
    batch, collation = timed(lambda: Batch.from_data_list(items))
    device = torch.device("cuda")
    gpu_batch, transfer = timed(lambda: batch.to(device, non_blocking=True))
    config = TrainConfig(architecture=args.family, hidden_dim=args.width,
                         node_features="hidden", edge_features="evidence_flow")
    model = build_model(config, item.x.shape[-1], item.edge_attr.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    torch.cuda.reset_peak_memory_stats()
    _, forward = timed(lambda: model(gpu_batch)[0])
    def backward():
        optimizer.zero_grad(set_to_none=True)
        score, _ = model(gpu_batch); score.square().mean().backward(); optimizer.step()
    _, train_step = timed(backward)
    result = {"family": args.family, "width": args.width, "batch_size": len(items),
              "nodes": int(batch.num_nodes), "edges": int(batch.num_edges),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "seconds": {"cold_first_item": cold, "warm_item": warm,
                          "item_feature_window": item_window, "collation": collation,
                          "host_to_device": transfer, "forward": forward,
                          "forward_backward_update": train_step},
              "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
