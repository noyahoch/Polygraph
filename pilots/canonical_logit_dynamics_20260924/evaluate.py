"""Canonical per-seed metrics and resumable paired source-photo uncertainty."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from .protocol import (METADATA, RECIPE, SEEDS, atomic_bytes, atomic_json, atomic_npz, campaign,
                       digest, evaluation_gate, frozen_json, lock, read, require_slurm, run_dir, sha256, verify)


def metrics(y, score):
    require_slurm()
    if len(np.unique(y)) != 2 or not np.isfinite(score).all():
        raise RuntimeError("Both canonical outcomes and finite scores are required")
    order = np.argsort(score, kind="stable")
    risks = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    return {"auroc": float(roc_auc_score(y, score)),
            "average_precision": float(average_precision_score(y, score)), "aurc": float(risks.mean())}


class WeightedAUC:
    """Exact score-tie grouped weighted AUROC; rankings stay fixed in this bootstrap."""
    def __init__(self, y, score):
        require_slurm()
        self.order = np.argsort(score, kind="stable")
        ordered = np.asarray(score)[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(ordered) != 0) + 1]
        self.y = np.asarray(y, dtype=np.float64)[self.order]

    def __call__(self, weights):
        weights = np.asarray(weights, dtype=np.float64)[self.order]
        positives = np.add.reduceat(weights * self.y, self.starts)
        negatives = np.add.reduceat(weights * (1 - self.y), self.starts)
        pos, neg = positives.sum(), negatives.sum()
        if not pos or not neg:
            return None
        return float(np.sum(positives * (np.cumsum(negatives) - 0.5 * negatives)) / (pos * neg))


def weighted_fixture_checks():
    require_slurm()
    y = np.asarray([0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    score = np.asarray([0.5, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0])
    groups = np.asarray([0, 0, 1, 2, 2, 2, 3])  # deliberately unequal view counts
    calculate = WeightedAUC(y, score)
    for multiplicity in ([1, 1, 1, 1], [0, 2, 1, 3], [3, 1, 0, 2]):
        weights = np.asarray(multiplicity)[groups]
        reference = roc_auc_score(np.repeat(y, weights), np.repeat(score, weights))
        np.testing.assert_allclose(calculate(weights), reference, atol=1e-12, rtol=0)
    if calculate(np.zeros(len(y))) is not None:
        raise RuntimeError("Undefined weighted AUROC was not retained")
    return {"weighted_auc_literal_duplication": True, "unequal_views": True, "ties": True}


def paired_bootstrap(directory, y, image_ids, polygraph, ld, identity):
    require_slurm()
    # RecordKey.group_id lexicographic ordering, identical to protocol, not numeric ID order.
    groups = sorted({f"test:{int(image_id)}" for image_id in image_ids})
    if len(groups) != 1998:
        raise RuntimeError("Canonical evaluation photo count changed")
    frozen_json(directory / "bootstrap_groups.json", {"groups": groups, "inputs": identity,
                 "draws": RECIPE["bootstrap_draws"], "seed": RECIPE["bootstrap_seed"]})
    positions = {group: i for i, group in enumerate(groups)}
    membership = np.asarray([positions[f"test:{int(i)}"] for i in image_ids], dtype=np.int64)
    rng = np.random.default_rng(RECIPE["bootstrap_seed"])
    calculators = [[WeightedAUC(y, matrix[:, i]) for i in range(len(SEEDS))] for matrix in (polygraph, ld)]
    output = []
    for draw in range(RECIPE["bootstrap_draws"]):
        multiplicities = np.bincount(rng.integers(0, len(groups), size=len(groups)), minlength=len(groups))
        draw_identity = {"inputs": identity, "draw_id": draw, "weights_sha256": digest(multiplicities.tolist())}
        path = directory / "draws" / f"draw_{draw:04d}.json"
        if path.exists():
            receipt = read(path)
            if receipt["identity"] != draw_identity:
                raise RuntimeError("Bootstrap resume inputs/multiplicities changed")
            payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            if receipt.get("receipt_sha256") != digest(payload) or receipt["multiplicities"] != multiplicities.tolist():
                raise RuntimeError("Bootstrap draw receipt changed")
        else:
            weights = multiplicities[membership]
            values = [[calculate(weights) for calculate in method] for method in calculators]
            valid = all(value is not None for method in values for value in method)
            differences = [left - right for left, right in zip(*values)] if valid else None
            receipt = {"identity": draw_identity, "valid": valid, "polygraph_auroc": values[0],
                       "ld_auroc": values[1], "seed_differences": differences,
                       "multiplicities": multiplicities.tolist(),
                       "mean_difference": float(np.mean(differences)) if valid else None}
            receipt["receipt_sha256"] = digest(receipt)
            atomic_json(path, receipt)
        output.append(receipt)
    invalid = [item["identity"]["draw_id"] for item in output if not item["valid"]]
    interval = None if invalid else np.quantile([item["mean_difference"] for item in output],
                                               [0.025, 0.975], method="linear").tolist()
    frozen_json(directory / "bootstrap_draw_manifest.json", {
        "inputs": identity, "groups_sha256": sha256(directory / "bootstrap_groups.json"),
        "files": {f"draws/draw_{draw:04d}.json": sha256(directory / "draws" / f"draw_{draw:04d}.json")
                  for draw in range(RECIPE["bootstrap_draws"])}})
    return {"draws": len(output), "seed": RECIPE["bootstrap_seed"], "interval_95": interval,
            "invalid_draw_ids": invalid, "complete": not invalid, "group_count": len(groups),
            "group_order_sha256": digest(groups), "conditional_on_fitted_models": True,
            "quantiles": [0.025, 0.975], "interpolation": "linear"}


def evaluate(root):
    require_slurm()
    root = Path(root).resolve()
    current = campaign(root)
    evaluation_gate(root)
    directory = root / "evaluation"
    reference = current["inputs"]["polygraph_scores"]
    verify(reference["path"], reference["sha256"])
    rows = [r for r in read(root / "records.json") if r["pool"] == "test"]
    metadata = {key: np.asarray([r[key] for r in rows], dtype=np.int64) for key in METADATA}
    if len(rows) != 17000:
        raise RuntimeError("Canonical evaluation count differs")
    with np.load(reference["path"], allow_pickle=False) as stored:
        for key in ("image_id", "source_id", "severity", "y"):
            np.testing.assert_array_equal(metadata[key], stored[key])
        polygraph = np.column_stack([stored[f"score_seed{seed}"] for seed in SEEDS])
    ld, identities = [], {"campaign_sha256": sha256(root / "campaign.json"), "polygraph_sha256": reference["sha256"],
                          "freeze_sha256": sha256(root / "evaluation_gate.json")}
    for seed in SEEDS:
        folder = run_dir(root, seed) / "predictions"
        receipt = read(folder / "complete.json")
        if (receipt.get("complete") is not True or receipt["identity"]["gate_sha256"] != identities["freeze_sha256"]
                or receipt["identity"]["campaign_sha256"] != identities["campaign_sha256"]
                or receipt["identity"]["seed"] != seed or receipt["identity"]["stage"] != "predict"):
            raise RuntimeError("Incomplete/unfrozen LD predictions")
        path = folder / "dev_eval.npz"
        verify(path, receipt["files"]["dev_eval.npz"])
        identities[f"seed{seed}"] = sha256(path)
        with np.load(path, allow_pickle=False) as stored:
            for key in METADATA:
                np.testing.assert_array_equal(metadata[key], stored[key])
            ld.append(stored["score"])
    ld = np.column_stack(ld)
    if polygraph.shape != (17000, 3) or ld.shape != polygraph.shape:
        raise RuntimeError("Three aligned seeds are required")
    with lock(directory, "evaluate"):
        if (directory / "complete.json").exists():
            complete = read(directory / "complete.json")
            if complete["inputs"] != identities:
                raise RuntimeError("Completed analysis inputs differ")
            for name, expected in complete["files"].items():
                verify(directory / name, expected)
            return read(directory / "report.json")
        checks = weighted_fixture_checks()
        results = {name: {str(seed): metrics(metadata["y"], matrix[:, i]) for i, seed in enumerate(SEEDS)}
                   for name, matrix in (("Polygraph", polygraph), ("LogitDynamics", ld))}
        summary = {name: {metric: {"mean": float(np.mean([item[metric] for item in values.values()])),
                                  "sd": float(np.std([item[metric] for item in values.values()], ddof=1))}
                          for metric in ("auroc", "average_precision", "aurc")} for name, values in results.items()}
        # Supplied historical reference is verified independently; mean metrics, never mean scores.
        np.testing.assert_allclose([summary["Polygraph"][m]["mean"] for m in ("auroc", "average_precision", "aurc")],
            [0.8939063783160323, 0.874839083136075, 0.20710602279320625], atol=1e-12, rtol=0)
        per_seed_difference = {str(seed): results["Polygraph"][str(seed)]["auroc"] - results["LogitDynamics"][str(seed)]["auroc"] for seed in SEEDS}
        interval = paired_bootstrap(directory, metadata["y"], metadata["image_id"], polygraph, ld, identities)
        primary = {"contrast": "Polygraph minus LogitDynamics AUROC", "estimate": float(np.mean(list(per_seed_difference.values()))),
                   "per_seed": per_seed_difference, **interval}
        report = {"complete": interval["complete"], "inputs": identities, "primary": primary,
                  "per_seed_metrics": results, "method_summary": summary, "seeds": list(SEEDS),
                  "records": len(rows), "photographs": len(np.unique(metadata["image_id"])),
                  "error_count": int(metadata["y"].sum()), "metric_checks": checks,
                  "limitations": ["Canonical benchmark was previously used for development; not an untouched-test claim.",
                      "Canonical records were selected using classifier correctness with constructed 50% error prevalence; AP/AURC describe this benchmark, not natural deployment prevalence.",
                      "Fresh LD auxiliary heads and probes use a different supervision allocation from Polygraph.",
                      "Fixed ViT-B adaptation, not full reproduction of the original paper search.",
                      "Photo bootstrap conditions on these fitted models; it does not include configuration-selection or future-seed variability.",
                      "An advantage between these complete recipes does not establish graph-topology necessity."]}
        atomic_npz(directory / "scores.npz", **metadata, polygraph=polygraph, logit_dynamics=ld, seeds=np.asarray(SEEDS))
        atomic_json(directory / "report.json", report)
        text = "# Canonical Polygraph versus LogitDynamics\n\n"
        text += "17,000 identical canonical views, 1,998 source photographs; seeds 1/2/7. Original scan error labels are preserved.\n\n"
        text += "| Method | Mean AUROC | Seed SD | Mean AP | Mean AURC |\n|---|---:|---:|---:|---:|\n"
        for name, values in summary.items():
            text += f"| {name} | {values['auroc']['mean']:.8f} | {values['auroc']['sd']:.8f} | {values['average_precision']['mean']:.8f} | {values['aurc']['mean']:.8f} |\n"
        text += f"\nMean within-seed Polygraph − LD AUROC: {primary['estimate']:.8f}; conditional 95% paired-photo interval: {interval['interval_95']}.\n\n"
        text += "\n".join("- " + value for value in report["limitations"]) + "\n"
        atomic_bytes(directory / "REPORT.md", text.encode())
        if interval["complete"]:
            atomic_json(directory / "complete.json", {"complete": True, "inputs": identities,
                         "files": {name: sha256(directory / name) for name in ("scores.npz", "report.json", "REPORT.md", "bootstrap_groups.json", "bootstrap_draw_manifest.json")},
                         "job_id": os.environ["SLURM_JOB_ID"]})
        else:
            raise RuntimeError("Undefined fixed bootstrap draws retained; interval/report marked incomplete")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.root)), flush=True)


if __name__ == "__main__":
    main()
