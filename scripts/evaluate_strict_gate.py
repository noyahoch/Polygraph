#!/usr/bin/env python3
"""Five-fold group-aware strict mixture gate over fixed output/internal experts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluate_combiners import load_aligned, neural_candidate, predict_artifact
from polygraph.training.evaluate import detector_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-val", type=Path, required=True)
    parser.add_argument("--output-test", type=Path, required=True)
    parser.add_argument("--internal-val", type=Path, required=True)
    parser.add_argument("--internal-test", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--detector-seed", type=int, required=True)
    parser.add_argument("--gate-seed", type=int, default=20260830)
    parser.add_argument("--method-name", default="strict_output_M5_gate")
    args = parser.parse_args()

    val_scores, val_meta = load_aligned([args.output_val, args.internal_val])
    test_scores, test_meta = load_aligned([args.output_test, args.internal_test])
    val_names = [str(np.load(p, allow_pickle=False)["method_name"].item())
                 for p in (args.output_val, args.internal_val)]
    test_names = [str(np.load(p, allow_pickle=False)["method_name"].item())
                  for p in (args.output_test, args.internal_test)]
    if val_names != test_names:
        raise ValueError(f"validation/test method mismatch: {val_names} != {test_names}")
    x_val = np.column_stack([val_scores[name] for name in val_names]).astype(np.float32)
    x_test = np.column_stack([test_scores[name] for name in test_names]).astype(np.float32)
    y = val_meta["y"].astype(np.int8)
    groups = val_meta["image_id"]

    oof = np.empty(len(y), np.float32)
    test_predictions, test_alphas, folds = [], [], []
    splitter = GroupKFold(n_splits=5)
    for fold, (train_idx, held_idx) in enumerate(splitter.split(x_val, y, groups)):
        assert not set(groups[train_idx]) & set(groups[held_idx])
        _, artifact = neural_candidate(
            "gate", x_val[train_idx], y[train_idx], x_val[held_idx], y[held_idx],
            x_val[train_idx, 0], x_val[held_idx, 0],
            x_val[train_idx, 1], x_val[held_idx, 1], seed=args.gate_seed + fold)
        held_score, _ = predict_artifact(artifact, x_val[held_idx],
                                         x_val[held_idx, 0], x_val[held_idx, 1])
        test_score, alpha = predict_artifact(artifact, x_test, x_test[:, 0], x_test[:, 1])
        oof[held_idx] = held_score
        test_predictions.append(test_score); test_alphas.append(alpha)
        folds.append({"fold": fold, "train_records": int(len(train_idx)),
                      "validation_records": int(len(held_idx)),
                      "validation": detector_metrics(y[held_idx], held_score)})

    score = np.mean(test_predictions, axis=0)
    alpha = np.mean(test_alphas, axis=0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"scores_test_seed{args.detector_seed}.npz"
    fields = {key: test_meta[key] for key in
              ("y", "confidence", "margin", "image_id", "source_id", "severity", "store_index")}
    np.savez_compressed(path, score=score, **fields,
                        method_name=np.asarray(args.method_name),
                        plan_hash=test_meta["plan_hash"], seed=np.asarray(args.detector_seed))
    report = {"method": args.method_name, "detector_seed": args.detector_seed,
              "gate_seed": args.gate_seed, "inputs": val_names,
              "selection": "fixed mixture gate; five-fold base-image GroupKFold on meta_val",
              "oof_meta_validation": detector_metrics(y, oof), "folds": folds,
              "test": detector_metrics(test_meta["y"], score),
              "alpha": {"mean": float(alpha.mean()), "std": float(alpha.std()),
                        "correct_mean": float(alpha[test_meta["y"] == 0].mean()),
                        "incorrect_mean": float(alpha[test_meta["y"] == 1].mean()),
                        "confident_mean": float(alpha[test_meta["confidence"] >= .9].mean())},
              "score_file": str(path)}
    (args.out_dir / f"summary_seed{args.detector_seed}.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
