#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polygraph.data.sidecars import atomic_json_save, atomic_torch_save, build_manifest
from polygraph.data.storage import GraphShard
from polygraph.training.graph_stats import final_layer_graph_statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--layer", type=int, default=11)
    args = p.parse_args()
    manifest = json.loads((args.store_dir / "manifest.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for shard_index, (name, count) in enumerate(zip(manifest["shards"], manifest["shard_records"])):
        out = args.out_dir / f"graph_stats_{shard_index:05d}.pt"
        if out.exists():
            payload = torch.load(out, map_location="cpu")
            if int(payload.get("records", -1)) == int(count):
                total += int(count); continue
            raise RuntimeError(f"incompatible completed graph-stat shard: {out}")
        shard = GraphShard.load(args.store_dir / name)
        rows = []
        for offset in range(len(shard)):
            graph = shard.layer_graph(offset, args.layer)
            rows.append(final_layer_graph_statistics(graph.edge_index, graph.edge_attr,
                                                      shard.diagonals[offset, args.layer],
                                                      shard.num_tokens, shard.tau))
        atomic_torch_save({"features": torch.stack(rows), "records": len(shard),
                           "layer": args.layer}, out)
        total += len(shard)
        print(f"graph stats shard {shard_index}: {len(shard)} records", flush=True)
    sidecar = build_manifest(args.store_dir, manifest.get("model_id", "unknown"),
                             {"features": ["N", 161], "per_head": 13, "cross_head": 5},
                             "float32", sys.argv, layer=args.layer)
    atomic_json_save(sidecar, args.out_dir / "manifest.json")
    print(f"graph statistics extracted for {total} records")


if __name__ == "__main__":
    main()
