"""Paired fixed-baseline comparison; one prespecified photograph-cluster interval."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from pilots.final_comparison_20260916.evaluate import score_metrics
from pilots.layer_ensemble_20260914.evaluate import WeightedAUC
from .protocol import (METADATA, RECIPE, SEEDS, atomic_bytes, atomic_json, atomic_npz, campaign,
                       evaluation_gate, lock, read, require_slurm, run_dir, sha256, verify)


def paired_bootstrap(labels, image_ids, g_scores, ld_scores, *, draws=2000, rng_seed=20260919):
    """Within-seed AUROC differences averaged after pairing; all views share source weights."""
    require_slurm()
    labels, image_ids = np.asarray(labels), np.asarray(image_ids)
    g_scores, ld_scores = np.asarray(g_scores), np.asarray(ld_scores)
    if g_scores.shape != ld_scores.shape or g_scores.shape != (len(labels), len(SEEDS)):
        raise ValueError("Expected aligned [records,three seeds] score matrices")
    if not np.isfinite(g_scores).all() or not np.isfinite(ld_scores).all():
        raise ValueError("Bootstrap scores must be finite")
    groups, membership, frequency = np.unique(image_ids, return_inverse=True, return_counts=True)
    if not np.all(frequency == 9):
        raise ValueError("Source photographs must retain all nine views")
    calculators = [[WeightedAUC(labels, matrix[:, index]) for index in range(len(SEEDS))]
                   for matrix in (g_scores, ld_scores)]
    points = np.asarray([[calculate() for calculate in method] for method in calculators], dtype=float)
    if not np.isfinite(points).all():
        raise ValueError("Primary point AUROC is undefined")
    rng = np.random.default_rng(rng_seed)
    multiplicities = np.empty((draws, len(groups)), dtype=np.int32)
    seed_differences = np.full((draws, len(SEEDS)), np.nan)
    for draw in range(draws):
        counts = np.bincount(rng.integers(0, len(groups), size=len(groups)), minlength=len(groups))
        multiplicities[draw] = counts
        weights = counts[membership]
        for index in range(len(SEEDS)):
            left, right = calculators[0][index](weights), calculators[1][index](weights)
            if left is not None and right is not None:
                seed_differences[draw, index] = left - right
    differences = seed_differences.mean(axis=1)
    undefined = np.flatnonzero(~np.isfinite(differences)).tolist()
    result = {
        "contrast": "G_mean minus LogitDynamics", "estimate": float((points[0] - points[1]).mean()),
        "per_seed": {str(seed): float(points[0, i] - points[1, i]) for i, seed in enumerate(SEEDS)},
        "interval_95": None if undefined else np.quantile(differences, [0.025, 0.975], method="linear").tolist(),
        "undefined_draw_indices": undefined, "draws": draws, "seed": rng_seed,
        "group": "image_id", "views_per_photograph": 9, "shared_draws": True,
        "estimand": "mean within-seed paired AUROC difference", "conditional_on_fitted_models": True,
        "undefined_draw_policy": "retain without redrawing; withhold affected confidence interval",
    }
    return result, {"image_id": groups, "multiplicity": multiplicities,
                    "per_seed_differences": seed_differences, "mean_differences": differences}


def align_metadata(reference, candidate):
    for key in METADATA:
        if candidate[key].dtype != np.int64 or not np.array_equal(reference[key], candidate[key]):
            raise RuntimeError("Frozen scores have unpaired metadata: " + key)


def evaluate(root):
    require_slurm()
    root = Path(root).resolve()
    current = campaign(root)
    evaluation_gate(root)
    directory = root / "evaluation"
    baseline_path = Path(current["baseline_root"]) / "evaluation" / "scores.npz"
    verify(baseline_path, current["baseline_scores_sha256"])
    with np.load(baseline_path, allow_pickle=False) as archive:
        metadata = {key: archive[key] for key in METADATA}
        baseline = dict(zip(archive["score_keys"].tolist(), archive["scores"].T))
    roles = read(root / "role_map.json")["roles"]["dev_eval"]
    if (metadata["record_id"].tolist() != roles["record_ids"] or len(metadata["y"]) != 7200
            or len(np.unique(metadata["image_id"])) != 800):
        raise RuntimeError("Baseline does not match the exact 800-source development role")
    scores = {key: value for key, value in baseline.items()
              if key.split("/")[0] in ("G_mean", "S_mean", "O", "MSP", "entropy")}
    inputs = {"campaign_sha256": sha256(root / "campaign.json"),
              "gate_sha256": sha256(root / "evaluation_gate.json"), "baseline_sha256": sha256(baseline_path)}
    for seed in SEEDS:
        prediction_dir = run_dir(root, seed) / "predictions"
        receipt = read(prediction_dir / "complete.json")
        if (receipt.get("complete") is not True or receipt["identity"]["gate_sha256"] != inputs["gate_sha256"]):
            raise RuntimeError("Predictions must come from the common three-seed freeze")
        path = prediction_dir / "dev_eval.npz"
        verify(path, receipt["files"]["dev_eval.npz"])
        with np.load(path, allow_pickle=False) as archive:
            align_metadata(metadata, archive)
            scores[f"LogitDynamics/seed{seed}"] = archive["score"]
        inputs[f"LogitDynamics/seed{seed}"] = sha256(path)
    with lock(directory, "evaluate"):
        if (directory / "complete.json").exists():
            complete = read(directory / "complete.json")
            if complete["inputs"] != inputs:
                raise RuntimeError("Completed evaluation input identity changed")
            for name, expected in complete["files"].items():
                verify(directory / name, expected)
            return read(directory / "report.json")
        metrics = {key: score_metrics(metadata["y"], value)[0] for key, value in scores.items()}
        summaries = {}
        for method in ("G_mean", "LogitDynamics", "S_mean", "O", "MSP", "entropy"):
            keys = [method] if method in ("MSP", "entropy") else [f"{method}/seed{s}" for s in SEEDS]
            summaries[method] = {}
            for metric in ("auroc", "average_precision", "aurc"):
                values = [metrics[key][metric] for key in keys]
                summaries[method][metric] = {"mean": float(np.mean(values)),
                                             "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        primary, bootstrap = paired_bootstrap(metadata["y"], metadata["image_id"],
            np.stack([scores[f"G_mean/seed{s}"] for s in SEEDS], axis=1),
            np.stack([scores[f"LogitDynamics/seed{s}"] for s in SEEDS], axis=1),
            draws=RECIPE["bootstrap_draws"], rng_seed=RECIPE["bootstrap_seed"])
        keys = sorted(scores)
        atomic_npz(directory / "scores.npz", **metadata, score_keys=np.asarray(keys),
                   scores=np.stack([scores[key] for key in keys], axis=1))
        atomic_npz(directory / "bootstrap.npz", **bootstrap)
        histories = {str(seed): {stage: read(run_dir(root, seed) / stage / "history.json") for stage in ("heads", "probe")}
                     for seed in SEEDS}
        result = {"complete": True, "scope_id": current["scope_id"], "inputs": inputs,
                  "primary": primary, "method_summary": summaries, "per_seed_metrics": metrics,
                  "training_histories": histories, "records": 7200, "source_photographs": 800, "seeds": list(SEEDS),
                  "error_count": int(metadata["y"].sum()), "error_prevalence": float(metadata["y"].mean()),
                  "auxiliary_parameters_per_seed": 922800, "probe_parameters_per_seed": 86,
                  "total_learned_parameters_per_seed": 922886, "original_test_access": False,
                  "limitations": ["Development data were previously exposed; this is not an untouched-test confirmation.",
                      "This is a fixed-budget ViT-Base adaptation, not a reproduction of the ViT-Large paper search.",
                      "The confidence interval conditions on these fitted models and does not measure full training variability.",
                      "G_mean and LogitDynamics have different capacity and training allocation; this does not isolate topology.",
                      "Only the prespecified G_mean-minus-LogitDynamics AUROC contrast has an inferential interval."]}
        atomic_json(directory / "report.json", result)
        text = "# LogitDynamics frozen development comparison\n\n"
        text += "800 source photographs, all nine views each, seeds 7/17/27. Original test remains closed.\n\n"
        text += f"Frozen classifier error prevalence: {result['error_count']}/7200 ({result['error_prevalence']:.2%}).\n\n"
        text += "| Method | Mean AUROC | Seed SD | Mean average precision | Mean AURC |\n|---|---:|---:|---:|---:|\n"
        for name, summary in summaries.items():
            sd = summary["auroc"]["sd"]
            text += (f"| {name} | {summary['auroc']['mean']:.6f} | {sd if sd is not None else 'N/A'} | "
                     f"{summary['average_precision']['mean']:.6f} | {summary['aurc']['mean']:.6f} |\n")
        text += (f"\nPrespecified mean paired G_mean − LogitDynamics ΔAUROC: {primary['estimate']:.6f}; "
                 f"ordinary 95% photograph-cluster bootstrap interval: {primary['interval_95']}. "
                 "All 2,000 draws use shared photograph multiplicities across methods and seeds. "
                 "Average precision is sklearn AP, not trapezoidal PR area. Seed means average metrics, never predictions.\n\n")
        text += "\n".join("- " + value for value in result["limitations"]) + "\n"
        atomic_bytes(directory / "REPORT.md", text.encode())
        files = ("scores.npz", "bootstrap.npz", "report.json", "REPORT.md")
        atomic_json(directory / "complete.json", {"complete": True, "inputs": inputs,
                    "files": {name: sha256(directory / name) for name in files}, "job_id": os.environ["SLURM_JOB_ID"]})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.root)
    print(json.dumps({"complete": result["complete"], "primary": result["primary"]}), flush=True)


if __name__ == "__main__":
    main()
