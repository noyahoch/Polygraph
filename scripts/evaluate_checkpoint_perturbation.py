#!/usr/bin/env python3
"""Inference-only topology/attribute perturbation of a fixed M5 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from polygraph.data.splits import SplitPlan
from polygraph.data.storage import AttentionGraphDataset, GraphStore
from polygraph.training.evaluate import detector_metrics, slice_masks
from polygraph.training.train import collect, load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--store", type=Path, default=Path("data/graph_dataset/store"))
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--mode", choices=["target_permute", "shuffle_attr"], required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    device = torch.device("cuda")
    model, config = load_checkpoint(args.checkpoint, device)
    plan, store = SplitPlan.load(args.plan), GraphStore(args.store)
    dataset = AttentionGraphDataset(store, config.layers, plan.splits["test"],
                                    hidden_dir=args.store.parent / "hidden12",
                                    edge_features="evidence_flow",
                                    message_stats_dir=args.store.parent / "sidecars/message_stats_l11",
                                    rewire_mode=args.mode, rewire_cache_dir=args.cache)
    pred = collect(model, dataset, device, config.batch_size, include_alignment=True)
    seed = int(config.seed); method = f"fixed_M5_inference_{args.mode}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.out_dir / f"scores_test_seed{seed}.npz"
    np.savez_compressed(score_path, score=pred["logit"],
                        **{k: pred[k] for k in ("y", "confidence", "margin", "image_id",
                                                "source_id", "severity", "store_index")},
                        method_name=np.asarray(method),
                        plan_hash=np.asarray(hashlib.sha256(args.plan.read_bytes()).hexdigest()),
                        seed=np.asarray(seed))
    seen = sorted({key.source for key in plan.splits["train"]})
    metrics = {name: detector_metrics(pred["y"][mask], pred["logit"][mask])
               for name, mask in slice_masks(pred, seen).items() if mask.any()}
    report = {"method": method, "interpretation": "distribution-shift sensitivity; model not adapted",
              "checkpoint": str(args.checkpoint), "metrics": metrics, "score_file": str(score_path)}
    (args.out_dir / f"summary_seed{seed}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
