"""Small saved-results EDA, explicitly authorized for local processing after the run."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from pilots.layer_ensemble_20260914.combine import head_scores
from pilots.layer_ensemble_20260914.evaluate import WeightedAUC, sigmoid
from .collect import ARMS, read, verify

SEEDS = (7, 17, 27)
LAYERS = (3, 6, 9, 12)
METADATA = ("record_id", "image_id", "source_id", "severity", "split_id", "y", "label", "pred")
METHODS = ("learned_last_only", "learned_stack", "fixed_probability_mean",
           "raw_block2", "raw_block5", "raw_block8", "raw_block11")
SOURCES = ("Clean", "Gaussian noise", "Motion blur", "Fog", "JPEG compression")
CONDITIONS = ((0, 0),) + tuple((source, severity) for source in range(1, 5) for severity in (3, 5))
OLD_METHODS = (
    ("full_graph", "Full graph MPNN", "graph"),
    ("full_set", "Full node/edge set", "no-message-passing"),
    ("full_endpoint", "Endpoint set", "no-message-passing"),
    ("full_rewired", "Partially rewired graph (descriptive)", "graph"),
    ("raw_graph", "Attention-only graph", "graph"),
    ("raw_set", "Attention-only set", "no-message-passing"),
    ("logit", "Full-logit MLP", "output-only"),
    ("msp", "ViT confidence (MSP)", "confidence"),
    ("entropy", "ViT output entropy", "confidence"),
)
CONTRASTS = (
    ("full_graph_minus_full_set", "Full graph minus full set"),
    ("full_graph_minus_full_endpoint", "Full graph minus endpoint set"),
    ("raw_graph_minus_raw_set", "Attention-only graph minus set"),
    ("full_graph_minus_full_rewired", "Original minus partial rewiring (descriptive)"),
    ("full_gap_minus_raw_gap", "Full-feature gap minus attention-only gap (secondary)"),
)


def assert_close(actual, expected, label, tolerance=1e-12):
    if actual is None or expected is None or not np.isclose(actual, expected, atol=tolerance, rtol=0):
        raise RuntimeError(f"Saved-result metric mismatch: {label}: {actual} != {expected}")


def load_scores(path):
    with zipfile.ZipFile(path) as archive:
        if sum(member.file_size for member in archive.infolist()) > 16 * 1024 * 1024:
            raise RuntimeError("Result arrays exceed the lightweight uncompressed size ceiling")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(METADATA + METHODS):
            raise RuntimeError("Unexpected result-only score schema")
        result = {name: archive[name].copy() for name in archive.files}
    if any(value.shape != (7200,) or not np.isfinite(value).all() for value in result.values()):
        raise RuntimeError("Expected finite, aligned 7,200-row result arrays")
    if (not np.isin(result["y"], (0, 1)).all()
            or not np.array_equal(result["y"], result["pred"] != result["label"])
            or len(np.unique(result["y"])) != 2 or len(np.unique(result["record_id"])) != 7200):
        raise RuntimeError("Invalid or duplicated frozen-classifier outcomes")
    photos, counts = np.unique(result["image_id"], return_counts=True)
    if len(photos) != 800 or not np.all(counts == 9):
        raise RuntimeError("Source photographs must retain exactly nine views each")
    if set(zip(result["source_id"].tolist(), result["severity"].tolist())) != set(CONDITIONS):
        raise RuntimeError("The fixed nine-condition mixture changed")
    return result


def rank_percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Ranks require a nonempty finite vector")
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.repeat((starts + ends) / (2 * len(values)), ends - starts)
    return ranks


def rank_correlation(columns):
    ranks = np.column_stack([rank_percentiles(values) for values in columns])
    ranks -= ranks.mean(axis=0)
    products = ranks.T @ ranks
    norms = np.sqrt(np.diag(products))
    matrix = []
    for i in range(len(columns)):
        row = []
        for j in range(len(columns)):
            denominator = norms[i] * norms[j]
            row.append(float(np.clip(products[i, j] / denominator, -1, 1)) if denominator else None)
        matrix.append(row)
    return matrix


def roc_points(labels, scores, max_points=257):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if (labels.ndim != 1 or labels.shape != scores.shape or not np.isin(labels, (0, 1)).all()
            or not np.isfinite(scores).all() or len(np.unique(labels)) != 2):
        raise ValueError("ROC requires aligned finite scores and both binary outcomes")
    if max_points < 2:
        raise ValueError("At least two display points are required")
    order = np.argsort(-scores, kind="stable")
    ordered, outcomes = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered) - 1]
    positives = np.cumsum(outcomes)[ends]
    negatives = (ends + 1) - positives
    points = np.column_stack((np.r_[0, negatives / (labels == 0).sum()],
                              np.r_[0, positives / (labels == 1).sum()]))
    full_auc = float(np.trapezoid(points[:, 1], points[:, 0]))
    assert_close(full_auc, WeightedAUC(labels, scores)(), "ROC/tie parity")
    if len(points) > max_points:
        points = points[np.unique(np.linspace(0, len(points) - 1, max_points, dtype=int))]
    return points.tolist()


def rank_histograms(scores):
    edges = np.linspace(0, 1, 21)
    result = {}
    for method in ("learned_stack", "learned_last_only"):
        ranks = rank_percentiles(scores[method])
        result[method] = {
            name: (np.histogram(ranks[scores["y"] == label], bins=edges)[0]
                   / (scores["y"] == label).sum()).tolist()
            for label, name in ((0, "correct"), (1, "error"))
        }
    return {"edges": edges.tolist(), "methods": result}


def condition_rows(scores):
    rows = []
    for source, severity in CONDITIONS:
        mask = (scores["source_id"] == source) & (scores["severity"] == severity)
        labels = scores["y"][mask]
        if int(mask.sum()) != 800 or len(np.unique(scores["image_id"][mask])) != 800:
            raise RuntimeError("Each condition must contain the same 800 source photographs")
        stack = WeightedAUC(labels, scores["learned_stack"][mask])()
        last = WeightedAUC(labels, scores["learned_last_only"][mask])()
        mean = WeightedAUC(labels, scores["fixed_probability_mean"][mask])()
        defined = stack is not None and last is not None
        rows.append({
            "label": SOURCES[source] if source == 0 else f"{SOURCES[source]} / severity {severity}",
            "source_id": source, "severity": severity, "records": int(mask.sum()), "photos": 800,
            "errors": int(labels.sum()), "error_rate": float(labels.mean()),
            "stack": stack, "last": last, "mean": mean,
            "delta": stack - last if defined else None, "defined": defined,
            "undefined_reason": None if defined else "Only one outcome class; AUROC undefined",
        })
    return rows


def source_error_counts(scores):
    _, membership = np.unique(scores["image_id"], return_inverse=True)
    errors = np.bincount(membership, weights=scores["y"]).astype(int)
    return {"counts": np.bincount(errors, minlength=10).tolist(), "photos": 800}


def seed_data(raw, seed, report, reference):
    root = raw / f"layers/seed{seed}"
    scores = load_scores(root / "evaluation/scores.npz")
    if reference is not None and any(not np.array_equal(scores[name], reference[name]) for name in METADATA):
        raise RuntimeError("Cannot pair results from different records, outcomes or ordering")
    saved = report["per_seed"][str(seed)]
    values = {name: float(saved["metrics"][name]["auroc"]) for name in METHODS}
    for name in METHODS:
        assert_close(WeightedAUC(scores["y"], scores[name])(), values[name], f"seed{seed}/{name}")
    logits = np.column_stack([scores["raw_" + arm] for arm in ARMS])
    for head_name, method in (("stack", "learned_stack"), ("last_only", "learned_last_only")):
        head = read(root / "heads" / (head_name + ".json"))
        if (head["seed"] != seed or head["training_role"] != "meta" or not head["converged"]
                or not head["serialization_audit"]["passed"]):
            raise RuntimeError("Invalid stored meta head")
        if not np.allclose(head_scores(head, logits), scores[method], atol=1e-12, rtol=1e-12):
            raise RuntimeError("Stored head does not reconstruct saved detector scores")
    if not np.allclose(sigmoid(logits).mean(axis=1), scores["fixed_probability_mean"], atol=1e-12, rtol=0):
        raise RuntimeError("The descriptive mean must average sigmoid scores, not raw logits")
    histories = []
    for arm, layer in zip(ARMS, LAYERS):
        directory = root / "runs" / arm / f"seed{seed}"
        history, complete = read(directory / "history.json"), read(directory / "complete.json")
        if [row["epoch"] for row in history] != list(range(1, 21)):
            raise RuntimeError("History must contain exactly 20 completed epochs")
        best = max(history, key=lambda row: row["checkpoint_auroc"])["epoch"]
        if complete["best_epoch"] != best:
            raise RuntimeError("Selected checkpoint is not the earliest greatest checkpoint AUROC")
        histories.append({
            "arm": arm, "layer": layer, "best_epoch": best,
            "epochs": [{"epoch": row["epoch"], "loss": row["training_loss"],
                        "checkpoint_auroc": row["checkpoint_auroc"],
                        "seconds": row["training_checkpoint_seconds"]} for row in history],
        })
    head = read(root / "heads/stack.json")
    entry = {
        "seed": seed, "status": saved["status"], "metrics": values,
        "delta": saved["primary"]["estimate"], "interval_95": saved["primary"]["interval_95"],
        "history": histories,
        "head_coefficients": [{"layer": layer, "coefficient": coefficient}
                              for layer, coefficient in zip(LAYERS, head["model"]["coef"])],
        "eda": {
            "correlation": {"labels": ["L3", "L6", "L9", "L12"],
                            "matrix": rank_correlation([scores["raw_" + arm] for arm in ARMS])},
            "roc": {name: roc_points(scores["y"], scores[name]) for name in
                    ("learned_stack", "learned_last_only", "fixed_probability_mean")},
            "rank_histograms": rank_histograms(scores),
            "conditions": condition_rows(scores), "source_error_counts": source_error_counts(scores),
        },
    }
    assert_close(entry["delta"], values["learned_stack"] - values["learned_last_only"], "paired seed delta")
    return entry, scores


def topology_data(raw):
    summary = read(raw / "september10/summary.json")
    bootstrap = read(raw / "september10/bootstrap.json")
    if summary["seeds"] != [1, 2, 7, 17, 27] or summary["status"] != "complete":
        raise RuntimeError("Historical experiment seed/status identity changed")
    rows = [row for row in summary["seed_results_and_conditions"] if row["scope"] == "mixture"]
    baselines = []
    for method, label, family in OLD_METHODS:
        selected = [row for row in rows if row["arm"] == method]
        seeds = [row["seed"] for row in selected]
        expected = [None] if method in ("msp", "entropy") else summary["seeds"]
        if len(seeds) != len(expected) or set(seeds) != set(expected):
            raise RuntimeError("Missing or repeated historical baseline seed: " + method)
        if any(row["records"] != 7200 or row["source_photographs"] != 800 for row in selected):
            raise RuntimeError("Historical baseline evaluation population differs")
        by_seed = {row["seed"]: row["auroc"] for row in selected}
        aucs = [by_seed[seed] for seed in expected]
        baselines.append({
            "method": method, "label": label, "family": family,
            "mean_auroc": float(np.mean(aucs)),
            "sample_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None,
            "per_seed": [{"seed": seed, "auroc": by_seed[seed]} for seed in expected],
        })
    contrasts = []
    for name, label in CONTRASTS:
        saved = bootstrap["comparisons"]["mixture"][name]
        contrasts.append({"label": label, "estimate": saved["estimate"],
                          "interval_95": saved["interval_95"],
                          "primary": name == "full_graph_minus_full_set"})
    primary = summary["primary"]
    graph = next(row for row in baselines if row["method"] == "full_graph")
    control = next(row for row in baselines if row["method"] == "full_set")
    assert_close(graph["mean_auroc"] - control["mean_auroc"], primary["estimate"], "old primary contrast")
    return {
        "title": "September 10: graph versus non-message-passing baselines",
        "population": "Previously evaluated held-out 800-photo test group / 7,200 records; five fitted seeds. Different from the current development group.",
        "primary": {**primary, "practical_margin": summary["practical_margin"]},
        "seeds": summary["seeds"], "baselines": baselines, "contrasts": contrasts,
        "errors": rows[0]["errors"], "correct": rows[0]["correct"],
        "limitations": [
            "Historical result tables only; no old test predictions were downloaded or rescored.",
            "Never rank these absolute AUROCs against the current layer study: different evaluation photographs and protocol.",
            "Whiskers on baseline means are descriptive sample SD across fitted seeds, not confidence intervals.",
            "The original paired bootstrap intervals are conditional on five fitted seeds; no new intervals were computed.",
            "Partial rewiring changed about 50.4% of edges, below its 80% criterion; topology removal was not established.",
            "The full-feature primary interval crosses the registered 0.005 practical-benefit margin.",
        ],
    }


def csv_rows(layers, topology):
    result = []

    def add(study, scope, seed, method, metric, value, status):
        result.append(dict(study=study, scope=scope, seed=seed, method=method,
                           metric=metric, value=value, status=status))

    for seed in layers["seeds"]:
        for name, value in seed["metrics"].items():
            add("layers_20260916", "dev_eval", seed["seed"], name, "auroc", value, seed["status"])
        add("layers_20260916", "dev_eval", seed["seed"], "stack_minus_last", "delta_auroc", seed["delta"], seed["status"])
        for condition in seed["eda"]["conditions"]:
            add("layers_20260916", condition["label"], seed["seed"], "stack_minus_last",
                "delta_auroc", condition["delta"], "posthoc_descriptive_no_ci")
    for name in ("primary", "all3"):
        for metric in ("estimate", "sample_sd"):
            add("layers_20260916", name, "/".join(map(str, layers[name]["seeds"])), "stack_minus_last",
                metric, layers[name][metric], "registered" if name == "primary" else "descriptive_includes_late7")
    for baseline in topology["baselines"]:
        for row in baseline["per_seed"]:
            add("topology_20260910", "historical_test_mixture", row["seed"], baseline["method"],
                "auroc", row["auroc"], "historical_complete")
    return result


def build(raw):
    started = time.perf_counter()
    raw = Path(raw).resolve()
    report = verify(raw)
    download = read(raw / "download.json")
    if (report["primary_seeds"] != [17, 27] or len(report["included_seeds"]) != 3
            or set(report["included_seeds"]) != set(SEEDS)
            or report["test_evaluated"] is not False or report["base_epochs"] != 20):
        raise RuntimeError("Unexpected current experiment reporting contract")
    seeds, reference = [], None
    for seed in SEEDS:
        entry, scores = seed_data(raw, seed, report, reference)
        if reference is None:
            reference = scores
        seeds.append(entry)
    layers = {
        "title": "September 14-16: four-layer stack versus one final-layer GNN",
        "population": "Same 800 development photographs / 7,200 records across seeds; not a new test.",
        "protocol_status": "Seeds 17/27 complete on time; seed 7 remains a historical late diagnostic.",
        "primary": {"estimate": report["primary"]["estimate"], "interval_95": report["primary"]["interval_95"],
                    "sample_sd": report["primary"]["statistics"]["delta_auroc"]["sample_sd"], "seeds": [17, 27],
                    "statistics": report["primary"]["statistics"]},
        "all3": {"estimate": report["all3_descriptive"]["estimate"],
                 "interval_95": report["all3_descriptive"]["interval_95"],
                 "sample_sd": report["all3_descriptive"]["statistics"]["delta_auroc"]["sample_sd"],
                 "seeds": [7, 17, 27], "statistics": report["all3_descriptive"]["statistics"]},
        "seeds": seeds, "roles": {"base_train": 1600, "checkpoint": 400, "meta": 400, "dev_eval": 800},
        "records": 7200, "photos": 800, "errors": int(reference["y"].sum()),
        "correct": int((reference["y"] == 0).sum()),
        "limitations": report["interpretation_limits"],
    }
    selected_deltas = [row["delta"] for row in seeds if row["seed"] in (17, 27)]
    assert_close(float(np.mean(selected_deltas)), layers["primary"]["estimate"], "new primary mean")
    assert_close(float(np.std(selected_deltas, ddof=1)), layers["primary"]["sample_sd"], "new sample SD")
    for group in ("primary", "all3"):
        for method in METHODS:
            values = [row["metrics"][method] for row in seeds if row["seed"] in layers[group]["seeds"]]
            recorded = layers[group]["statistics"][method]
            assert_close(float(np.mean(values)), recorded["mean"], f"{group}/{method}/mean")
            assert_close(float(np.std(values, ddof=1)), recorded["sample_sd"], f"{group}/{method}/SD")
    topology = topology_data(raw)
    off_diagonal = [row["eda"]["correlation"]["matrix"][i][j]
                    for row in seeds for i in range(4) for j in range(i + 1, 4)]
    mean_gaps = [row["metrics"]["fixed_probability_mean"] - row["metrics"]["learned_stack"] for row in seeds]
    negative_slices = [f"seed {row['seed']}: {condition['label']}" for row in seeds
                       for condition in row["eda"]["conditions"]
                       if condition["delta"] is not None and condition["delta"] < 0]
    insights = [
        f"The registered mean gain over the final-layer GNN is {layers['primary']['estimate']:+.5f} AUROC in seeds 17/27; all three observed paired differences are positive.",
        f"Fixed averaging is slightly above the learned stack in every observed seed: differences range from {min(mean_gaps):+.5f} to {max(mean_gaps):+.5f} AUROC. This is descriptive, not a tested superiority claim.",
        f"Layer-score Spearman correlations range from {min(off_diagonal):.3f} to {max(off_diagonal):.3f} across the three saved runs. High correlation means overlapping rankings, not that the scores are interchangeable.",
        "The historical baseline chart includes real non-graph controls, but it uses a different evaluation group. It cannot establish that today's ensemble beats those baselines.",
    ]
    if negative_slices:
        insights.append("The pooled gain is not uniform across every condition: negative descriptive differences appear for "
                        + "; ".join(negative_slices) + ". No condition-specific significance claim is made.")
    notes = [
        "This results-only local analysis was explicitly requested after all models, heads and formal outputs were frozen.",
        "Original AUROCs and paired confidence intervals come from the completed Slurm reports; every displayed saved-score AUROC was checked against its report.",
        "Correlation, ROC/rank distributions and condition slices are post-hoc descriptive EDA. No additional inference, bootstrap, model selection or tuning was performed.",
        "Condition slices share the same photographs. Pooled AUROC includes cross-condition comparisons and is not the average of within-condition AUROCs.",
        "Source IDs are corruption families; source photographs are image_id. Nine variants are not nine independent photographs.",
        "ROC coordinates are thinned to at most 257 points for display; AUROCs use the full unthinned scores and tie-aware calculation.",
        "Rank histograms are normalized separately within correct/error outcomes. Rank is not a calibrated error probability.",
        "Checkpoint curves are the original checkpoint-selection role, not dev_eval or test. Selected checkpoints remain unchanged.",
        "Standardized logistic coefficients are fitted predictive weights with correlated inputs, not causal layer importance.",
        "No additional Slurm jobs, classifier/detector inference, original-test access or external web requests are needed for this local view.",
    ]
    result = {
        "schema_version": 1, "title": "Polygraph · results explorer",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "provenance": {
            "download_bytes": download["bytes"], "download_files": len(download["files"]),
            "bundle_paths": ["raw/download.json", "raw/layers/report.json", "raw/september10/summary.json",
                             "raw/september10/results.csv", "results.json", "results.csv"],
            "notes": ["All imported result files are checksum-bound to their original completion receipts.",
                      "The bundle contains no model weights, optimizer state, feature tensors or images."],
        },
        "studies": {"layers": layers, "topology": topology}, "insights": insights, "analysis_notes": notes,
        "csv_rows": csv_rows(layers, topology),
        "build_seconds": time.perf_counter() - started,
    }
    json.dumps(result, allow_nan=False)
    return result


def save(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("study", "scope", "seed", "method", "metric", "value", "status"))
        writer.writeheader()
        writer.writerows(data["csv_rows"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = build(args.raw)
    save(data, args.out)
    print(json.dumps({"output": str(args.out), "build_seconds": data["build_seconds"],
                      "download_bytes": data["provenance"]["download_bytes"],
                      "insights": data["insights"]}, indent=2))


if __name__ == "__main__":
    main()
