#!/usr/bin/env python3
"""End-to-end validation of the graph dataset on disk (legacy fixed-count format).

Checks the extracted graphs against the scan records and the manifest, confirms the
group-disjoint and stratification properties on the data as written rather than as
intended, and reports the MSP baseline that any graph detector has to beat.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader

from legacy_dataset import InMemoryLayerDataset
from lightweight_attention_experiments import ReadoutModel
from polygraph.config import ALL_SOURCES
from polygraph.records import ScanRecordStore

SPLITS = ("train", "val", "test")


def summarize(name: str, labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    if len(np.unique(labels)) <= 1:
        return {"detector": name, "n": int(labels.size), "auroc": float("nan"), "auprc": float("nan")}
    return {
        "detector": name,
        "n": int(labels.size),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs-dir", default="data/graph_dataset/graphs")
    parser.add_argument("--scan-file", default="data/graph_dataset/scan_records.jsonl")
    parser.add_argument("--layer", type=int, default=11, help="0-based ViT layer; 11 is the final layer")
    args = parser.parse_args()

    root = Path(args.graphs_dir)
    failures: List[str] = []
    datasets = {}
    for split in SPLITS:
        datasets[split] = InMemoryLayerDataset(root / split, layer_positions=[args.layer])
        print(f"{split}: {len(datasets[split])} records")

    scan = {record.key.as_tuple(): record for record in ScanRecordStore(Path(args.scan_file))}

    groups: Dict[str, set] = {}
    for split, dataset in datasets.items():
        meta = dataset.meta
        y = meta["y_err"].numpy()
        source = meta["source_id"].numpy()
        severity = meta["severity"].numpy()
        base = meta["base_index"].numpy()
        confidence = meta["confidence"].numpy()

        wrong, correct = int((y > 0.5).sum()), int((y <= 0.5).sum())
        if wrong != correct:
            failures.append(f"{split} is not 50/50: {wrong} wrong vs {correct} correct")

        space = np.where(source == ALL_SOURCES.index("clean_train"), 1, 0)
        groups[split] = set(zip(space.tolist(), base.tolist()))

        cells: Dict[tuple, List[int]] = defaultdict(lambda: [0, 0])
        for s, v, label in zip(source.tolist(), severity.tolist(), y.tolist()):
            cells[(s, v)][0 if label > 0.5 else 1] += 1
        unbalanced = {k: v for k, v in cells.items() if v[0] != v[1]}
        if unbalanced:
            failures.append(f"{split} has {len(unbalanced)} cells that are not class-balanced")

        mismatched = 0
        for index in range(0, len(dataset), max(len(dataset) // 500, 1)):
            key = (ALL_SOURCES[int(source[index])], int(severity[index]), int(base[index]))
            record = scan.get(key)
            if record is None or int(record.correct) != int(y[index] < 0.5):
                mismatched += 1
            elif abs(record.confidence - float(confidence[index])) > 1e-5:
                mismatched += 1
        if mismatched:
            failures.append(f"{split} has {mismatched} records disagreeing with the scan")

        print(
            f"  {split}: wrong={wrong} correct={correct} cells={len(cells)} "
            f"unbalanced_cells={len(unbalanced)} scan_mismatches={mismatched}"
        )

    for left in SPLITS:
        for right in SPLITS:
            if left < right and groups[left] & groups[right]:
                failures.append(f"base-image leak between {left} and {right}: {len(groups[left] & groups[right])}")

    sample = datasets["test"][0]
    if tuple(sample.x.shape) != (197, 16) or tuple(sample.edge_attr.shape) != (100, 12):
        failures.append(f"unexpected graph shapes: x={tuple(sample.x.shape)} edge_attr={tuple(sample.edge_attr.shape)}")
    if int((sample.x[:, 2] > 0.5).sum()) != 1:
        failures.append("CLS indicator does not select exactly one node")

    batch = next(iter(DataLoader(datasets["test"], batch_size=16, shuffle=False)))
    model = ReadoutModel(in_dim=16, edge_dim=12, hidden_dim=32, layers=2, dropout=0.15, readout="cls_gated")
    logits, _ = model(batch)
    if tuple(logits.shape) != (16,):
        failures.append(f"ReadoutModel returned {tuple(logits.shape)}")

    print("\nMSP baseline on the new test split (the bar a graph detector must beat):")
    meta = datasets["test"].meta
    y = meta["y_err"].numpy()
    msp = 1.0 - meta["confidence"].numpy()
    source = meta["source_id"].numpy()
    severity = meta["severity"].numpy()
    report = [summarize("MSP, all test records", y, msp)]
    clean = source == ALL_SOURCES.index("clean_test")
    report.append(summarize("MSP, clean images only", y[clean], msp[clean]))
    for level in (1, 3, 5):
        mask = severity == level
        if mask.any():
            report.append(summarize(f"MSP, corruption severity {level}", y[mask], msp[mask]))
    for row in report:
        print(f"  {row['detector']:34s} n={row['n']:6d}  AUROC={row['auroc']:.4f}  AUPRC={row['auprc']:.4f}")

    payload = {"msp_baseline": report, "splits": {s: len(d) for s, d in datasets.items()}}
    out = root.parent / "validation_report.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("\nAll dataset validation checks passed.")


if __name__ == "__main__":
    main()
