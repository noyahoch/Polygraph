#!/usr/bin/env python3
"""Create aligned final tables, severity slices, complementarity, and group bootstraps."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from polygraph.training.evaluate import detector_metrics
from polygraph.training.research_eval import paired_group_bootstrap, verify_score_registry

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs/topology_depth_last4_20260905"
PRIOR = ROOT / "runs/research_20260830/combiners/strict"


def load(path):
    return {key: value for key, value in np.load(path, allow_pickle=False).items()}


def flagged(score):
    mask = np.zeros(len(score), bool); mask[np.argsort(score, kind="stable")[-len(score)//2:]] = True
    return mask


def blind(y, score, reference):
    own, ref = flagged(score), flagged(reference)
    errors = y == 1; blind_spot = errors & ~ref
    return {"errors_caught": int((own & errors).sum()),
            "errors_reference_misses": int((own & errors & ~ref).sum()),
            "errors_reference_catches_method_misses": int((ref & errors & ~own).sum()),
            "blind_recovery": float((own & blind_spot).sum() / max(blind_spot.sum(), 1)),
            "flag_overlap": int((own & ref).sum())}


def runtime_for(path):
    total = 0.0
    ledger = RUN / "ledger.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row = json.loads(line)
            if path.parent.name in row.get("experiment_id", ""):
                total += float(row.get("runtime_seconds", 0))
    return total


def parameter_count(path):
    checkpoint = path.parent / (path.name.replace("scores_test_", "model_").replace(".npz", ".pt"))
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["state_dict"]
        history = payload.get("history", [])
        peak = max((row.get("peak_gpu_memory_mb") or 0 for row in history), default=None)
        best_epoch = max(history, key=lambda row: row["val_auroc"])["epoch"] if history else None
        return (sum(value.numel() for value in state.values()), str(checkpoint.relative_to(ROOT)),
                peak, best_epoch)
    return None, None, None, None


def references(plan):
    root = PRIOR / plan
    return {"output": root / "output", "M5": root / "M5", "gate": root / "gate"}


def all_score_paths():
    paths = list(RUN.glob("**/scores_test_seed*.npz"))
    for plan in ("main", "weather"):
        for directory in references(plan).values(): paths += list(directory.glob("scores_test_seed*.npz"))
    return sorted(set(paths))


def plan_of(path):
    text = str(path)
    return "weather" if "/weather/" in text or "/weather_" in text else "main"


def output_path(plan, seed):
    path = references(plan)["output"] / f"scores_test_seed{seed}.npz"
    if not path.exists(): path = references(plan)["output"] / "scores_test_seed7.npz"
    return path


def main():
    final = RUN / "final"; final.mkdir(parents=True, exist_ok=True)
    rows, severity, artifacts = [], {}, []
    for path in all_score_paths():
        data, plan = load(path), plan_of(path)
        if not {"score", "y", "confidence", "image_id", "store_index"} <= set(data): continue
        seed = int(data.get("seed", np.asarray(0)).item())
        method = str(data.get("method_name", np.asarray(path.parent.name)).item())
        output = load(output_path(plan, seed))
        verify_score_registry([output_path(plan, seed), path])
        y, score, msp = data["y"], data["score"], 1 - data["confidence"]
        metric = detector_metrics(y, score)
        confident = data["confidence"] >= .9
        params, checkpoint, peak, best_epoch = parameter_count(path)
        msp_blind, output_blind = blind(y, score, msp), blind(y, score, output["score"])
        row = {"method": method, "artifact_name": path.parent.name, "plan": plan, "seed": seed,
               "AUROC": metric.get("auroc"), "AUPRC": metric.get("auprc"), "AURC": metric.get("aurc"),
               "risk05": metric.get("risk@0.5"), "risk08": metric.get("risk@0.8"),
               "risk09": metric.get("risk@0.9"),
               "confident_AUROC": detector_metrics(y[confident], score[confident]).get("auroc"),
               "blindspot_MSP": msp_blind["blind_recovery"],
               "blindspot_best_output": output_blind["blind_recovery"],
               "spearman_MSP": float(spearmanr(score, msp).statistic),
               "spearman_best_output": float(spearmanr(score, output["score"]).statistic),
               "runtime_seconds": runtime_for(path), "params": params, "checkpoint": checkpoint,
               "peak_gpu_memory_mb": peak, "best_epoch": best_epoch,
               "score_file": str(path.relative_to(ROOT))}
        rows.append(row); artifacts.append(row["score_file"])
        severity[row["score_file"]] = {
            "all": metric,
            "confidence_ge_0.9": detector_metrics(y[confident], score[confident]),
            **{f"severity_{level}": detector_metrics(y[data["severity"] == level],
                                                       score[data["severity"] == level])
               for level in range(1, 6) if (data["severity"] == level).any()}}
    fields = list(rows[0]) if rows else []
    for filename in ("model_comparison.csv", "experiment_table.csv"):
        with (final / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(rows)
    (final / "severity_metrics.json").write_text(json.dumps(severity, indent=2) + "\n")

    # Choose gates by OOF meta-validation only; test metrics cannot enter this decision.
    gate_candidates = {}
    for summary in RUN.glob("gates/*/summary_seed*.json"):
        if summary.parent.name.startswith("weather_"):
            continue
        report = json.loads(summary.read_text())
        gate_candidates.setdefault(summary.parent.name, []).append(
            report["oof_meta_validation"]["auroc"])
    gate_selection = {"criterion": "mean OOF meta_val AUROC only", "test_metrics_used": False,
                      "candidates": {k: float(np.mean(v)) for k, v in gate_candidates.items()}}
    gate_selection["selected"] = max(gate_selection["candidates"], key=gate_selection["candidates"].get) \
        if gate_selection["candidates"] else None
    (final / "gate_selection.json").write_text(json.dumps(gate_selection, indent=2) + "\n")
    for plan in ("main", "weather"):
        candidates = {}
        pattern = "weather_*" if plan == "weather" else "*"
        for directory in RUN.joinpath("gates").glob(pattern):
            if plan == "main" and directory.name.startswith("weather_"): continue
            summaries = list(directory.glob("summary_seed*.json"))
            if summaries:
                candidates[directory] = np.mean([json.loads(p.read_text())["oof_meta_validation"]["auroc"]
                                                 for p in summaries])
        if candidates:
            chosen = max(candidates, key=candidates.get)
            payload = {"selected_by": np.asarray("mean OOF meta_val AUROC"),
                       "method_name": np.asarray(chosen.name)}
            for score_file in chosen.glob("scores_test_seed*.npz"):
                seed = int(score_file.stem.rsplit("seed", 1)[1]); data = load(score_file)
                payload[f"score_seed{seed}"] = data["score"]
                for key in ("y", "confidence", "image_id", "source_id", "severity", "store_index"):
                    payload.setdefault(key, data[key])
            np.savez_compressed(final / f"best_scores_{plan}.npz", **payload)

    bootstraps = {}
    selection = json.loads((final / "selection_manifest.json").read_text())
    comparisons = []
    for seed in (7, 1, 2):
        output_file = output_path("main", seed); output = load(output_file)
        comparisons.append((f"output_seed{seed}_vs_MSP", output_file, None))
        for label, candidate in (
            ("M5", references("main")["M5"] / f"scores_test_seed{seed}.npz"),
            (selection["controls"]["S1"], RUN / "controls" / selection["controls"]["S1"] / f"scores_test_seed{seed}.npz"),
            (selection["controls"]["S2"], RUN / "controls" / selection["controls"]["S2"] / f"scores_test_seed{seed}.npz")):
            if candidate.exists(): comparisons.append((f"{label}_seed{seed}_vs_output", candidate, output_file))
    if gate_selection["selected"]:
        for path in (RUN / "gates" / gate_selection["selected"]).glob("scores_test_seed*.npz"):
            seed = int(path.stem.rsplit("seed", 1)[1])
            comparisons.append((f"{gate_selection['selected']}_seed{seed}_vs_output",
                                path, output_path("main", seed)))
    for name, candidate_path, reference_path in comparisons:
        candidate = load(candidate_path)
        reference_score = 1 - candidate["confidence"] if reference_path is None else load(reference_path)["score"]
        bootstraps[name] = paired_group_bootstrap(candidate["y"], candidate["score"], reference_score,
                                                   candidate["image_id"], repetitions=2000, seed=20260905)
    (final / "paired_bootstrap.json").write_text(json.dumps(bootstraps, indent=2) + "\n")
    summary = {"rows": len(rows), "selection": selection, "gate_selection": gate_selection,
               "severity_metrics": "severity_metrics.json", "bootstrap": "paired_bootstrap.json"}
    (final / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (final / "artifact_map.json").write_text(json.dumps({"artifacts": sorted(artifacts)}, indent=2) + "\n")
    mandatory = {name: any(name in str(p) for p in all_score_paths())
                 for name in (selection["controls"]["S1"], selection["controls"]["S2"],
                              "L0_token_trajectory", "L1_union_graph", "L1_SET_union_endpoint")}
    (final / "mandatory_completion.json").write_text(json.dumps(mandatory, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
