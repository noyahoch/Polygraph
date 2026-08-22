"""Evaluation: consumes checkpoints from training.py; re-runnable without retraining.

Metrics per slice (clean / seen corruptions / unseen sources / per severity): AUROC, AUPRC,
and selective prediction (risk-coverage, AURC). Baselines MSP and margin are computed on
the identical records, plus a train-fitted standardized logistic combiner — the Stage-5
lesson: unscaled features let the L2 penalty crush MSP and the combiner scored below it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from ..config import ALL_SOURCES, FAMILY_OF
from .train import collect, load_checkpoint


def detector_metrics(y: np.ndarray, score: np.ndarray) -> Dict[str, object]:
    """AUROC/AUPRC plus selective prediction: reject highest-scored records first;
    risk@c = ViT error rate in the kept c-fraction; AURC = mean risk over coverages."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    result = {"n": int(y.size), "positives": int(y.sum())}
    if len(np.unique(y)) < 2:
        return {**result, "auroc": None, "auprc": None}
    order = np.argsort(score, kind="stable")  # most trusted first
    cum_risk = np.cumsum(y[order]) / np.arange(1, y.size + 1)
    return {**result, "auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score)), "aurc": float(cum_risk.mean()),
            **{f"risk@{c}": float(cum_risk[int(np.ceil(c * y.size)) - 1]) for c in (0.5, 0.8, 0.9, 1.0)}}


def _combiner_features(pred: Dict[str, np.ndarray]) -> np.ndarray:
    conf = np.clip(pred["confidence"], 1e-6, 1 - 1e-6)
    return np.stack([pred["logit"], np.log((1 - conf) / conf)], axis=1)


def fit_combiner(train_pred: Dict[str, np.ndarray]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    combiner = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    combiner.fit(_combiner_features(train_pred), train_pred["y"])
    return combiner


def slice_masks(pred: Dict[str, np.ndarray], seen_sources: Sequence[str]) -> Dict[str, np.ndarray]:
    names = np.asarray([ALL_SOURCES[int(i)] for i in pred["source_id"]])
    families = np.asarray([FAMILY_OF[n] for n in names])
    seen = np.isin(names, list(seen_sources))
    masks = {"all": np.ones(len(names), bool), "clean": names == "clean_test",
             "seen_corruptions": seen & (names != "clean_test"),
             "unseen_sources": ~seen & (names != "clean_test"),
             "unseen_extra_family": families == "extra"}
    for severity in sorted(set(map(int, pred["severity"])) - {0}):
        masks[f"severity_{severity}"] = pred["severity"] == severity
    return masks


def evaluate_predictions(test_pred, train_pred, seen_sources) -> Dict[str, object]:
    combined = (fit_combiner(train_pred).predict_proba(_combiner_features(test_pred))[:, 1]
                if train_pred is not None else None)
    report = {}
    for name, mask in slice_masks(test_pred, seen_sources).items():
        if not mask.any():
            continue
        y = test_pred["y"][mask]
        report[name] = {"graph": detector_metrics(y, test_pred["logit"][mask]),
                        "msp": detector_metrics(y, 1 - test_pred["confidence"][mask]),
                        "margin": detector_metrics(y, -test_pred["margin"][mask])}
        if combined is not None:
            report[name]["msp_plus_graph"] = detector_metrics(y, combined[mask])
    return report


def evaluate_run(run_dir: Path, store_dir: Path, plan_path: Path, device,
                 seeds: Optional[Sequence[int]] = None) -> Dict[str, object]:
    """Evaluate every checkpoint in run_dir against a plan; aggregate across seeds.

    The graph view (layers/tau/top_k) comes from each checkpoint, so evaluation always
    sees the same graphs the model was trained on.
    """
    from ..data.splits import SplitPlan
    from ..data.storage import AttentionGraphDataset, GraphStore

    checkpoints = sorted(run_dir.glob("model_seed*.pt"))
    if seeds is not None:
        checkpoints = [p for p in checkpoints if int(p.stem.replace("model_seed", "")) in set(seeds)]
    assert checkpoints, f"no checkpoints in {run_dir}"
    plan = SplitPlan.load(plan_path)
    store = GraphStore(store_dir)
    seen_sources = sorted({k.source for k in plan.splits["train"]})

    per_seed = []
    baseline_features = None
    for path in checkpoints:
        model, config = load_checkpoint(path, device)
        datasets = {n: AttentionGraphDataset(store, config.layers, plan.splits[n],
                                             tau=config.tau, top_k=config.top_k)
                    for n in ("train", "val", "test")}
        train_pred = collect(model, datasets["train"], device, config.batch_size)
        test_pred = collect(model, datasets["test"], device, config.batch_size)
        report = evaluate_predictions(test_pred, train_pred, seen_sources)

        # Trained non-graph baselines, same protocol and seed as this checkpoint, so the
        # detector's advantage cannot be "it was trained" (features read once, reused).
        from .baselines import cls_mlp_scores, collect_features, output_scores
        if baseline_features is None:
            baseline_features = {n: collect_features(datasets[n]) for n in ("train", "val", "test")}
        seed = int(path.stem.replace("model_seed", ""))
        trained_baselines = {"output_lr": output_scores(baseline_features["train"],
                                                        baseline_features["test"], seed)}
        if "cls" in baseline_features["train"]:
            trained_baselines["cls_mlp"] = cls_mlp_scores(
                baseline_features["train"], baseline_features["val"], baseline_features["test"],
                seed, device)
        for name, scores in trained_baselines.items():
            for slice_name, mask in slice_masks(test_pred, seen_sources).items():
                if slice_name in report and mask.any():
                    report[slice_name][name] = detector_metrics(test_pred["y"][mask], scores[mask])
        (run_dir / f"report_{path.stem}.json").write_text(json.dumps(report, indent=2))
        per_seed.append(report)
        print(f"  {path.name}: test AUROC {report['all']['graph']['auroc']:.4f}, "
              f"AURC {report['all']['graph']['aurc']:.4f}", flush=True)

    summary: Dict[str, object] = {"checkpoints": [p.name for p in checkpoints],
                                  "plan": str(plan_path), "slices": {}}
    for slice_name in per_seed[0]:
        summary["slices"][slice_name] = {}
        for detector in per_seed[0][slice_name]:
            values = [s[slice_name][detector]["auroc"] for s in per_seed
                      if s[slice_name][detector]["auroc"] is not None]
            if values:
                summary["slices"][slice_name][detector] = dict(
                    auroc_mean=float(np.mean(values)), auroc_std=float(np.std(values)),
                    n=per_seed[0][slice_name][detector]["n"])
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
