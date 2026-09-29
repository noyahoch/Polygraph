#!/usr/bin/env python3
"""Inventory graph-store, split-plan, hidden-state, checkpoint, and score artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"_read_error": repr(exc)}


def key_tuple(value):
    if isinstance(value, dict):
        return (str(value.get("source")), int(value.get("severity", 0)),
                int(value.get("base_index", value.get("image_id", -1))))
    return tuple(value)


def plan_splits(plan):
    for container in (plan, plan.get("splits", {}) if isinstance(plan, dict) else {}):
        if isinstance(container, dict) and any(k in container for k in ("train", "val", "test")):
            return {k: container.get(k, []) for k in ("train", "val", "test")}
    return {k: [] for k in ("train", "val", "test")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--store-dir", type=Path, default=Path("data/graph_dataset/store"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    store = args.store_dir if args.store_dir.is_absolute() else root / args.store_dir
    report = {"root": str(root), "store_dir": str(store), "store_present": False}
    manifest_path, keys_path = store / "manifest.json", store / "store_keys.json"
    store_keys = []
    if manifest_path.exists() and keys_path.exists():
        manifest = load_json(manifest_path)
        store_keys = load_json(keys_path)
        encoded = json.dumps(store_keys, separators=(",", ":")).encode()
        report["store_present"] = True
        report["store"] = {
            "records": manifest.get("records", len(store_keys)),
            "shards": len(manifest.get("shards", [])),
            "shard_records": manifest.get("shard_records", []),
            "shard_sizes_bytes": {p.name: p.stat().st_size for p in sorted(store.glob("shard_*.pt"))},
            "tau": manifest.get("tau"),
            "model_id": manifest.get("model_id"),
            "num_tokens": manifest.get("num_tokens"),
            "num_layers": manifest.get("layer_count", manifest.get("layers")),
            "cls_embeddings": manifest.get("cls_embeddings"),
            "store_key_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    scan_labels = {}
    scan_path = root / "data/graph_dataset/scan_records.jsonl"
    if scan_path.exists():
        with scan_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (str(row["source"]), int(row["severity"]), int(row["base_index"]))
                scan_labels[key] = int(not bool(row["correct"]))
    report["scan_records"] = len(scan_labels)

    plans = []
    store_key_set = {key_tuple(value) for value in store_keys}
    for path in sorted((root / "data").glob("**/*plan*.json")) if (root / "data").exists() else []:
        plan = load_json(path)
        splits = plan_splits(plan)
        split_keys = {name: [key_tuple(v) for v in values] for name, values in splits.items()}
        bases = {name: {(k[0], k[2]) for k in values} for name, values in split_keys.items()}
        overlap = {f"{a}_{b}": len(bases[a] & bases[b]) for a, b in
                   (("train", "val"), ("train", "test"), ("val", "test"))}
        held_out = plan.get("config", {}).get("held_out", [])
        balance = {}
        missing_scan = 0
        for name, values in split_keys.items():
            labels = [scan_labels[k] for k in values if k in scan_labels]
            missing_scan += len(values) - len(labels)
            counts = Counter(labels)
            balance[name] = {"correct": counts[0], "wrong": counts[1]}
        plans.append({
            "path": str(path), "sizes": {k: len(v) for k, v in split_keys.items()},
            "held_out_sources": held_out,
            "base_image_overlap": overlap,
            "class_balance": balance,
            "missing_scan_records": missing_scan,
            "store_key_coverage": (sum(k in store_key_set
                                       for values in split_keys.values() for k in values)
                                   if store_keys else None),
        })
    report["plans"] = plans
    sidecar_root = root / "data/graph_dataset/sidecars"
    report["hidden_sidecars"] = []
    for path in sorted((root / "data/graph_dataset").glob("hidden*")) if (root / "data/graph_dataset").exists() else []:
        manifest = load_json(path / "manifest.json") if path.is_dir() else {}
        report["hidden_sidecars"].append({"path": str(path), "layer": manifest.get("layer")})
    if sidecar_root.exists():
        for path in sorted(sidecar_root.glob("*hidden*")):
            manifest = load_json(path / "manifest.json")
            report["hidden_sidecars"].append({"path": str(path), "layer": manifest.get("layer")})
    report["checkpoints"] = [str(p) for p in sorted((root / "runs").glob("**/model_seed*.pt"))]
    report["checkpoint_configs"] = [str(p) for p in sorted((root / "runs").glob("**/report_model_seed*.json"))]
    report["prediction_scores"] = [str(p) for p in sorted((root / "runs").glob("**/scores*.npz"))]
    usage = shutil.disk_usage(root)
    report["disk"] = {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0 if report["store_present"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
