#!/usr/bin/env python3
"""Group-disjoint exploratory combination of aligned detector scores.

All architecture selection happens on a held-out 30% of validation base-image
groups.  The test file is loaded only after the winning architecture is fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polygraph.training.evaluate import detector_metrics
from polygraph.training.research_eval import grouped_split


def load_aligned(paths: list[Path]) -> tuple[dict[str, np.ndarray], np.lib.npyio.NpzFile]:
    arrays = [np.load(path, allow_pickle=False) for path in paths]
    reference = arrays[0]
    reference_index = reference["store_index"]
    if len(np.unique(reference_index)) != len(reference_index):
        raise ValueError(f"duplicate store indices in {paths[0]}")
    scores = {}
    for path, item in zip(paths, arrays):
        index = item["store_index"]
        exact_metadata_order = (len(index) == len(reference_index) and all(
            np.array_equal(reference[field], item[field])
            for field in ("y", "image_id", "source_id", "severity")) and
            str(reference["plan_hash"].item()) == str(item["plan_hash"].item()))
        if np.array_equal(index, reference_index):
            order = np.arange(len(index))
        elif len(np.unique(index)) == len(index) and set(index.tolist()) == set(reference_index.tolist()):
            position = {int(value): i for i, value in enumerate(index)}
            order = np.asarray([position[int(value)] for value in reference_index])
        elif exact_metadata_order:
            # Compatibility for graph score files produced before GraphData fixed
            # PyG's auto-increment of the metadata field named store_index.  Four
            # independent identity arrays and the plan hash must match row-for-row.
            order = np.arange(len(index))
        else:
            raise ValueError(f"score registry record-set mismatch: {path}")
        for field in ("y", "image_id", "source_id", "severity"):
            if not np.array_equal(reference[field], item[field][order]):
                raise ValueError(f"score registry metadata mismatch in {field}: {path}")
        name = str(item["method_name"].item())
        if name in scores:
            raise ValueError(f"duplicate method name in registry: {name}")
        scores[name] = item["score"][order].astype(np.float32)
    return scores, reference


def _standardize(train: np.ndarray, *others: np.ndarray):
    mean, std = train.mean(0), train.std(0) + 1e-6
    return ((train - mean) / std,) + tuple((value - mean) / std for value in others), mean, std


def _calibrator(score: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=20260830)
    model.fit(score[:, None], y)
    return model


def logistic_candidate(x_train, y_train, x_val):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=20260830))
    model.fit(x_train, y_train)
    return model.predict_proba(x_val)[:, 1], model


class Residual(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.delta = nn.Sequential(nn.Linear(width, 16), nn.ReLU(), nn.Dropout(.1),
                                   nn.Linear(16, 16), nn.ReLU(), nn.Dropout(.1),
                                   nn.Linear(16, 1))
        nn.init.zeros_(self.delta[-1].weight); nn.init.zeros_(self.delta[-1].bias)

    def forward(self, x, output_logit):
        return output_logit + self.delta(x).view(-1)


class Gate(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(width, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, output_logit, internal_logit):
        alpha = torch.sigmoid(self.gate(x).view(-1))
        return (1 - alpha) * output_logit + alpha * internal_logit, alpha


def neural_candidate(kind, x_train, y_train, x_val, y_val, output_train, output_val,
                     internal_train, internal_val, seed=20260830):
    from sklearn.metrics import roc_auc_score
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    (xt, xv), mean, std = _standardize(x_train, x_val)
    output_cal = _calibrator(output_train, y_train)
    internal_cal = _calibrator(internal_train, y_train)
    ot = output_cal.decision_function(output_train[:, None]).astype(np.float32)
    ov = output_cal.decision_function(output_val[:, None]).astype(np.float32)
    it = internal_cal.decision_function(internal_train[:, None]).astype(np.float32)
    iv = internal_cal.decision_function(internal_val[:, None]).astype(np.float32)
    xt, xv = map(torch.from_numpy, (xt.astype(np.float32), xv.astype(np.float32)))
    yt = torch.from_numpy(y_train.astype(np.float32))
    ot, ov, it, iv = map(torch.from_numpy, (ot, ov, it, iv))
    model = Residual(x_train.shape[1]) if kind == "residual" else Gate(x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    positive = float(y_train.sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        (len(y_train) - positive) / max(positive, 1)))
    best, best_auc, stale = None, -np.inf, 0
    generator = torch.Generator().manual_seed(seed)
    for _epoch in range(100):
        model.train()
        for idx in torch.randperm(len(xt), generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            prediction = (model(xt[idx], ot[idx]) if kind == "residual" else
                          model(xt[idx], ot[idx], it[idx])[0])
            loss = loss_fn(prediction, yt[idx]); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            val = (model(xv, ov) if kind == "residual" else model(xv, ov, iv)[0]).numpy()
        auc = roc_auc_score(y_val, val)
        if auc > best_auc + .0005:
            best_auc, stale, best = auc, 0, deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 10:
                break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        val = (model(xv, ov) if kind == "residual" else model(xv, ov, iv)[0]).numpy()
    artifact = {"kind": kind, "model": model, "mean": mean, "std": std,
                "output_cal": output_cal, "internal_cal": internal_cal}
    return val, artifact


def predict_artifact(artifact, x, output, internal):
    kind = artifact["kind"]
    standardized = torch.from_numpy(((x - artifact["mean"]) / artifact["std"]).astype(np.float32))
    output_logit = torch.from_numpy(artifact["output_cal"].decision_function(output[:, None]).astype(np.float32))
    internal_logit = torch.from_numpy(artifact["internal_cal"].decision_function(internal[:, None]).astype(np.float32))
    artifact["model"].eval()
    with torch.no_grad():
        if kind == "residual":
            return artifact["model"](standardized, output_logit).numpy(), None
        score, alpha = artifact["model"](standardized, output_logit, internal_logit)
        return score.numpy(), alpha.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-val", type=Path, required=True)
    parser.add_argument("--output-test", type=Path, required=True)
    parser.add_argument("--internal-val", type=Path, required=True)
    parser.add_argument("--internal-test", type=Path, required=True)
    parser.add_argument("--extra-val", type=Path, nargs="*", default=[])
    parser.add_argument("--extra-test", type=Path, nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    val_paths = [args.output_val, args.internal_val, *args.extra_val]
    val_scores, val_meta = load_aligned(val_paths)
    ordered_names = [str(np.load(path, allow_pickle=False)["method_name"].item()) for path in val_paths]
    x = np.column_stack([val_scores[name] for name in ordered_names]).astype(np.float32)
    y = val_meta["y"].astype(np.int8)
    meta_train, meta_val = grouped_split(val_meta["image_id"], .70, 20260830)
    assert not set(val_meta["image_id"][meta_train]) & set(val_meta["image_id"][meta_val])
    output, internal = x[:, 0], x[:, 1]

    candidates, artifacts = {}, {}
    candidates["logistic"], artifacts["logistic"] = logistic_candidate(
        x[meta_train], y[meta_train], x[meta_val])
    for kind in ("residual", "gate"):
        candidates[kind], artifacts[kind] = neural_candidate(
            kind, x[meta_train], y[meta_train], x[meta_val], y[meta_val], output[meta_train],
            output[meta_val], internal[meta_train], internal[meta_val])
    validation = {name: detector_metrics(y[meta_val], score) for name, score in candidates.items()}
    selected = max(validation, key=lambda name: (validation[name]["auroc"], -validation[name]["aurc"]))

    # Deliberately load test arrays only after the architecture is fixed.
    test_paths = [args.output_test, args.internal_test, *args.extra_test]
    test_scores, test_meta = load_aligned(test_paths)
    test_names = [str(np.load(path, allow_pickle=False)["method_name"].item()) for path in test_paths]
    if test_names != ordered_names:
        raise ValueError(f"validation/test method mismatch: {ordered_names} != {test_names}")
    test_x = np.column_stack([test_scores[name] for name in test_names]).astype(np.float32)
    if selected == "logistic":
        test_score = artifacts[selected].predict_proba(test_x)[:, 1]
        alpha = None
    else:
        test_score, alpha = predict_artifact(artifacts[selected], test_x, test_x[:, 0], test_x[:, 1])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_hash = str(test_meta["plan_hash"].item())
    score_path = args.out_dir / "scores_test_seed20260830.npz"
    fields = {key: test_meta[key] for key in
              ("y", "confidence", "margin", "image_id", "source_id", "severity", "store_index")}
    np.savez_compressed(score_path, score=test_score, **fields,
                        method_name=np.asarray(f"combiner:{selected}"),
                        plan_hash=np.asarray(plan_hash), seed=np.asarray(20260830))
    report = {"selected_on": "group_disjoint_meta_validation", "selected": selected,
              "inputs": ordered_names, "meta_train_records": int(len(meta_train)),
              "meta_validation_records": int(len(meta_val)), "validation": validation,
              "test": detector_metrics(test_meta["y"], test_score),
              "score_file": str(score_path)}
    if alpha is not None:
        report["alpha"] = {"mean": float(alpha.mean()), "std": float(alpha.std()),
                           "correct_mean": float(alpha[test_meta["y"] == 0].mean()),
                           "incorrect_mean": float(alpha[test_meta["y"] == 1].mean()),
                           "confident_mean": float(alpha[test_meta["confidence"] >= .9].mean())}
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
